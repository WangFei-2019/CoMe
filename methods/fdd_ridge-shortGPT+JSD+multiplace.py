import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging

from utils.data_utils import get_trainloaders
from utils.eval_utils import load_and_eval_ppl

import torch
import torch.nn.functional as F

def diagnose_ppl_jsd_conflict(golden_logits, comp_logits, golden_hiddens, comp_hiddens, input_ids, tokenizer):
    """
    微观诊断探针：定位 PPL 飙升的罪魁祸首
    """
    # 1. 准备 Causal LM 的 Shifted Logits 和 Labels
    # 对于 Llama 等自回归模型，第 i 个 token 的 logit 预测的是第 i+1 个 token
    shift_labels = input_ids[..., 1:].contiguous().view(-1)
    shift_gold_logits = golden_logits[..., :-1, :].contiguous().view(-1, golden_logits.size(-1))
    shift_comp_logits = comp_logits[..., :-1, :].contiguous().view(-1, comp_logits.size(-1))
    
    # 2. 计算 Token 级别的交叉熵损失 (Cross Entropy)
    ce_gold = F.cross_entropy(shift_gold_logits, shift_labels, reduction='none')
    ce_comp = F.cross_entropy(shift_comp_logits, shift_labels, reduction='none')
    
    # 计算损失差值：正数表示补偿后预测变得更差了
    ce_diff = ce_comp - ce_gold
    
    # 3. 特征模长 (L2 Norm) 与方向 (Cosine Similarity) 分析
    # 对齐形状并计算最后一个 Token 之前的隐藏状态
    h_gold = golden_hiddens[..., :-1, :].contiguous().view(-1, golden_hiddens.size(-1))
    h_comp = comp_hiddens[..., :-1, :].contiguous().view(-1, comp_hiddens.size(-1))
    
    norm_gold = h_gold.norm(dim=-1)
    norm_comp = h_comp.norm(dim=-1)
    norm_ratio = (norm_comp / norm_gold).mean().item()
    
    # 4. 抓取“退化最严重”的 Top-K Tokens
    top_k = 15
    worst_indices = torch.topk(ce_diff, top_k).indices
    
    print("\n" + "="*80)
    print("🔍 [深度诊断] JSD 与 PPL 背离现象分析报告")
    print("="*80)
    print(f"📉 全局特征模长收缩比 (Comp Norm / Gold Norm): {norm_ratio:.4f}")
    if norm_ratio < 0.9:
        print("   ⚠️ 警告：特征模长严重收缩！这会导致 Logits 变平缓，使得模型失去自信，极大拉高 PPL！")
    elif norm_ratio > 1.1:
        print("   ⚠️ 警告：特征模长爆炸！这会触发 Logits 极端尖锐化！")
        
    print(f"\n🚨 退化最严重的 Top-{top_k} 个 Token (PPL 飙升的罪魁祸首):")
    print(f"{'Target Token':<15} | {'Gold Pred (Prob)':<20} | {'Comp Pred (Prob)':<20} | {'Δ CE Loss':<10}")
    print("-" * 80)
    
    for idx in worst_indices:
        target_token_id = shift_labels[idx].item()
        target_word = tokenizer.decode([target_token_id]).replace('\n', '\\n')
        
        # 获取两者的 Top-1 预测及概率
        gold_probs = F.softmax(shift_gold_logits[idx], dim=-1)
        comp_probs = F.softmax(shift_comp_logits[idx], dim=-1)
        
        gold_top1_id = torch.argmax(gold_probs).item()
        comp_top1_id = torch.argmax(comp_probs).item()
        
        gold_top1_word = tokenizer.decode([gold_top1_id]).replace('\n', '\\n')
        comp_top1_word = tokenizer.decode([comp_top1_id]).replace('\n', '\\n')
        
        gold_prob_val = gold_probs[target_token_id].item()
        comp_prob_val = comp_probs[target_token_id].item()
        
        diff_val = ce_diff[idx].item()
        
        print(f"[{target_word:<13}] | {gold_top1_word:<10} ({gold_prob_val:.2%}) | {comp_top1_word:<10} ({comp_prob_val:.2%}) | +{diff_val:.3f}")
    print("="*80 + "\n")

# ==========================================
# 🎛️ 递进式校准控制器 (在此处开关您想要调整的矩阵)
# ==========================================
TUNE_CONFIG = {
    "tune_kqv": False,       # 是否校准 L+1 的 q_proj, k_proj, v_proj
    "tune_attn_o": True,    # 是否校准 L+1 的 o_proj
    "tune_ffn_in": False,    # 是否校准 L+1 的 gate_proj, up_proj
    "tune_ffn_o": True,     # 是否校准 L+1 的 down_proj
    "damping_factor": 0.8   # 目标阻尼系数 (防方差爆炸)
}

class SurgeryHooks:
    def __init__(self):
        self.handles = []
        self.data = {}
    def clear(self): self.data.clear()
    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

def compute_jsd(logits_p, logits_q):
    p = F.softmax(logits_p.float(), dim=-1)
    q = F.softmax(logits_q.float(), dim=-1)
    m = 0.5 * (p + q)
    p = torch.clamp(p, min=1e-8)
    q = torch.clamp(q, min=1e-8)
    m = torch.clamp(m, min=1e-8)
    kl_p = F.kl_div(m.log(), p, reduction='none').sum(dim=-1)
    kl_q = F.kl_div(m.log(), q, reduction='none').sum(dim=-1)
    return (0.5 * (kl_p + kl_q)).mean().item()

class FDDEvaluator:
    def __init__(self, model, dataloader, num_batches=4):
        self.valid_inputs = []
        self.golden_logits = []
        self.golden_hiddens = []
        self.device = model.device
        logging.info("📊 提取 FDD 黃金基準線 (Golden Baseline)...")
        model.eval()
        with torch.no_grad():
            for i in range(num_batches):
                inputs = dataloader.input_ids[:, (i * model.seqlen):((i+1) * model.seqlen)].to(self.device)
                self.valid_inputs.append(inputs)
                outputs = model(inputs, output_hidden_states=True, use_cache=False)
                self.golden_logits.append(outputs.logits.cpu())
                self.golden_hiddens.append(outputs.hidden_states[-1].cpu())
        torch.cuda.empty_cache()

    def evaluate(self, model):
        model.eval()
        jsd_list, cos_list = [], []
        with torch.no_grad():
            for i, inputs in enumerate(self.valid_inputs):
                outputs = model(inputs, output_hidden_states=True, use_cache=False)
                jsd = compute_jsd(self.golden_logits[i], outputs.logits.cpu())
                jsd_list.append(jsd)
                cos_sim = F.cosine_similarity(self.golden_hiddens[i].float(), outputs.hidden_states[-1].cpu().float(), dim=-1).mean().item()
                cos_list.append(cos_sim)
        return np.mean(jsd_list), np.mean(cos_list)
    
    def run_diagnosis(self, model, tokenizer):
        """调用微观探针，仅使用第一个 Batch 防止显存爆炸"""
        model.eval()
        with torch.no_grad():
            inputs = self.valid_inputs[0] # [B, S]
            # 把 Golden 基线拉回到 GPU 以便运算
            gold_logits = self.golden_logits[0].to(self.device)
            gold_hiddens = self.golden_hiddens[0].to(self.device)
            
            # 获取补偿后模型的当前状态
            outputs = model(inputs, output_hidden_states=True, use_cache=False)
            comp_logits = outputs.logits
            comp_hiddens = outputs.hidden_states[-1]
            
            # 触发诊断打印
            diagnose_ppl_jsd_conflict(
                gold_logits, comp_logits, 
                gold_hiddens, comp_hiddens, 
                inputs, tokenizer
            )

def solve_ridge(Z, Y, device, reg_ratio=0.01):
    """通用的流式岭回归求解器，传入累加后的 ZtZ 和 ZtY"""
    D = Z.shape[0]
    norm = torch.trace(Z) / D
    lambda_reg = reg_ratio * norm if norm > 0 else 1e-4
    W_comp = torch.linalg.solve(Z + lambda_reg * torch.eye(D, device=device), Y)
    return W_comp

@torch.inference_mode()
def main_func(args, modelhander):
    logging.info(f"🚀 啟動 FDD 遞進式全鏈路補償算法 | 配置: {TUNE_CONFIG}")
    
    dataloader = get_trainloaders(args.calibration_dataset, tokenizer=modelhander.tokenizer, nsamples=args.nsamples, seed=args.seed, seqlen=modelhander.model.seqlen)
    compute_device = modelhander.model.device
    testenc = dataloader.input_ids
    seqlen = modelhander.model.seqlen
    nsamples = testenc.numel() // seqlen
    
    fdd_evaluator = FDDEvaluator(modelhander.model, dataloader, num_batches=4)
    iteration = 0
    total_to_prune = modelhander.model.config.num_hidden_layers - args.target_layers

    while modelhander.model.config.num_hidden_layers > args.target_layers:
        iteration += 1
        current_layer_num = modelhander.model.config.num_hidden_layers
        logging.info(f"\n{'='*70}\n🔄 輪次 {iteration}/{total_to_prune} | 當前層數: {current_layer_num}")
        
        # --- 沿用 ShortGPT 的选层逻辑 (快速且基于特征透明度) ---
        bi_scores = {l: 0.0 for l in range(1, current_layer_num - 1)}
        modelhander.model.eval()
        for i in tqdm(range(nsamples), desc=f"BI 掃描"):
            inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(compute_device)
            outputs = modelhander.model(inputs, output_hidden_states=True, use_cache=False)
            for l_idx in range(1, current_layer_num - 1):
                cos_sim = F.cosine_similarity(outputs.hidden_states[l_idx].float(), outputs.hidden_states[l_idx+1].float(), dim=-1).mean().item()
                bi_scores[l_idx] += (1.0 - cos_sim)
        
        prune_idx = min(bi_scores, key=bi_scores.get)
        comp_idx = prune_idx + 1
        logging.info(f"🎯 鎖定層: Layer {prune_idx} 將被切除，由 Layer {comp_idx} 進行遞進式補償。")

        layers = modelhander.model.model.layers
        L_p, L_c = layers[prune_idx], layers[comp_idx]
        

        def probe_current_state(stage_name, run_diag=False):
            logging.info(f"--- 🧪 探針: {stage_name} ---")
            bypass_hook = L_p.register_forward_hook(lambda m, i, o: (i[0],) + o[1:] if isinstance(o, tuple) else i[0])
            
            jsd, cos = fdd_evaluator.evaluate(modelhander.model)
            w_ppl = load_and_eval_ppl(modelhander.model, dataset="wiki2", tokenizer=modelhander.tokenizer)
            # c_ppl = load_and_eval_ppl(modelhander.model, dataset="c4", tokenizer=modelhander.tokenizer)

            # 如果开启了诊断，就在这里运行深层探针
            if run_diag:
                fdd_evaluator.run_diagnosis(modelhander.model, modelhander.tokenizer)
                
            bypass_hook.remove()
            logging.info(f"➤ JSD: {jsd:.6f} | Cos: {cos:.4f} | Wiki2: {w_ppl:.2f} \n") # | C4: {c_ppl:.2f}

        probe_current_state("[0] 直接剪除 (無補償)", run_diag=True)

        hooks = SurgeryHooks()
        num_calib = min(nsamples, 32)
        damp = TUNE_CONFIG["damping_factor"]

        # ---------------------------------------------------------------------
        # Step 1: 提取原模型的 Golden Targets (所有需要的原版激活值)
        # ---------------------------------------------------------------------
        orig_targets = {
            'Q': [], 'K': [], 'V': [], 'Op_attn': [], 'Oc_attn': [],
            'Gate': [], 'Up': [], 'Op_ffn': [], 'Oc_ffn': []
        }
        for i in tqdm(range(num_calib), desc="提取原模型目標特徵"):
            inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(compute_device)
            hooks.handles.extend([
                # 拿 L_c 原本应该输出的 Q,K,V (即输入是 X_mid 算出来的)
                L_c.self_attn.q_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Q': o.detach().cpu()})),
                L_c.self_attn.k_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'K': o.detach().cpu()})),
                L_c.self_attn.v_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'V': o.detach().cpu()})),
                # 拿 L_p 和 L_c 的输出，用于融合
                L_p.self_attn.o_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Op_attn': o.detach().cpu()})),
                L_c.self_attn.o_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Oc_attn': o.detach().cpu()})),
                # 拿 L_c 原本应该输出的 Gate, Up (即输入是 X_mid_ffn 算出来的)
                L_c.mlp.gate_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Gate': o.detach().cpu()})),
                L_c.mlp.up_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Up': o.detach().cpu()})),
                # 拿 FFN 融合目标
                L_p.mlp.down_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Op_ffn': o.detach().cpu()})),
                L_c.mlp.down_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Oc_ffn': o.detach().cpu()}))
            ])
            modelhander.model(inputs, use_cache=False)
            hooks.remove()
            
            for key in orig_targets.keys():
                orig_targets[key].append(hooks.data[key])
            hooks.clear()

        # 定义一个帮助函数，用于在这个阶段拦截输入 Z
        def collect_Z_and_solve(proj_layer, target_list, desc):
            D_in, D_out = proj_layer.weight.shape[1], proj_layer.weight.shape[0]
            ZtZ, ZtY = torch.zeros((D_in, D_in), device=compute_device), torch.zeros((D_in, D_out), device=compute_device)
            for i in tqdm(range(num_calib), desc=desc):
                inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(compute_device)
                hooks.handles.extend([
                    # 让 L_p 透明化
                    L_p.register_forward_hook(lambda m, i, o: (i[0],) + o[1:] if isinstance(o, tuple) else i[0]),
                    # 拦截目标层的输入
                    proj_layer.register_forward_hook(lambda m, inp, o: hooks.data.update({'Z': inp[0].detach().cpu()}))
                ])
                modelhander.model(inputs, use_cache=False)
                hooks.remove()
                
                Z = hooks.data['Z'].view(-1, D_in).float().to(compute_device)
                Y = target_list[i].view(-1, D_out).float().to(compute_device)
                ZtZ += Z.t() @ Z; ZtY += Z.t() @ Y
                hooks.clear()
            
            new_W = solve_ridge(ZtZ, ZtY, compute_device)
            proj_layer.weight.data = new_W.t().to(dtype=proj_layer.weight.dtype, device=proj_layer.weight.device)

        # ---------------------------------------------------------------------
        # Step 2: 递进式校准流程开始
        # ---------------------------------------------------------------------
        if TUNE_CONFIG["tune_kqv"]:
            # 注意：这三个可以并行拦截，因为它们的输入是同一个
            D_in = L_c.self_attn.q_proj.weight.shape[1]
            ZtZ = torch.zeros((D_in, D_in), device=compute_device)
            ZtY_q = torch.zeros((D_in, L_c.self_attn.q_proj.weight.shape[0]), device=compute_device)
            ZtY_k = torch.zeros((D_in, L_c.self_attn.k_proj.weight.shape[0]), device=compute_device)
            ZtY_v = torch.zeros((D_in, L_c.self_attn.v_proj.weight.shape[0]), device=compute_device)
            
            for i in tqdm(range(num_calib), desc="校准 KQV"):
                inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(compute_device)
                hooks.handles.extend([
                    L_p.register_forward_hook(lambda m, i, o: (i[0],) + o[1:] if isinstance(o, tuple) else i[0]),
                    # K, Q, V 共享同一个输入 Z
                    L_c.self_attn.q_proj.register_forward_hook(lambda m, inp, o: hooks.data.update({'Z': inp[0].detach().cpu()}))
                ])
                modelhander.model(inputs, use_cache=False)
                hooks.remove()
                Z = hooks.data['Z'].view(-1, D_in).float().to(compute_device)
                ZtZ += Z.t() @ Z
                ZtY_q += Z.t() @ orig_targets['Q'][i].view(-1, ZtY_q.shape[1]).float().to(compute_device)
                ZtY_k += Z.t() @ orig_targets['K'][i].view(-1, ZtY_k.shape[1]).float().to(compute_device)
                ZtY_v += Z.t() @ orig_targets['V'][i].view(-1, ZtY_v.shape[1]).float().to(compute_device)
                hooks.clear()
                
            L_c.self_attn.q_proj.weight.data = solve_ridge(ZtZ, ZtY_q, compute_device).t().to(
                dtype=L_c.self_attn.q_proj.weight.dtype, 
                device=L_c.self_attn.q_proj.weight.device
            )
            L_c.self_attn.k_proj.weight.data = solve_ridge(ZtZ, ZtY_k, compute_device).t().to(
                dtype=L_c.self_attn.k_proj.weight.dtype, 
                device=L_c.self_attn.k_proj.weight.device
            )
            L_c.self_attn.v_proj.weight.data = solve_ridge(ZtZ, ZtY_v, compute_device).t().to(
                dtype=L_c.self_attn.v_proj.weight.dtype, 
                device=L_c.self_attn.v_proj.weight.device
            )
            probe_current_state("[1] 已校准: KQV")

        if TUNE_CONFIG["tune_attn_o"]:
            target_attn_o = [(Op + Oc) * damp for Op, Oc in zip(orig_targets['Op_attn'], orig_targets['Oc_attn'])]
            collect_Z_and_solve(L_c.self_attn.o_proj, target_attn_o, "校准 Attn O_proj")
            probe_current_state("[2] 已校准: KQV + O")

        if TUNE_CONFIG["tune_ffn_in"]:
            D_in = L_c.mlp.gate_proj.weight.shape[1]
            ZtZ = torch.zeros((D_in, D_in), device=compute_device)
            ZtY_g = torch.zeros((D_in, L_c.mlp.gate_proj.weight.shape[0]), device=compute_device)
            ZtY_u = torch.zeros((D_in, L_c.mlp.up_proj.weight.shape[0]), device=compute_device)
            
            for i in tqdm(range(num_calib), desc="校准 Gate/Up"):
                inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(compute_device)
                hooks.handles.extend([
                    L_p.register_forward_hook(lambda m, i, o: (i[0],) + o[1:] if isinstance(o, tuple) else i[0]),
                    L_c.mlp.gate_proj.register_forward_hook(lambda m, inp, o: hooks.data.update({'Z': inp[0].detach().cpu()}))
                ])
                modelhander.model(inputs, use_cache=False)
                hooks.remove()
                Z = hooks.data['Z'].view(-1, D_in).float().to(compute_device)
                ZtZ += Z.t() @ Z
                ZtY_g += Z.t() @ orig_targets['Gate'][i].view(-1, ZtY_g.shape[1]).float().to(compute_device)
                ZtY_u += Z.t() @ orig_targets['Up'][i].view(-1, ZtY_u.shape[1]).float().to(compute_device)
                hooks.clear()
                
            L_c.mlp.gate_proj.weight.data = solve_ridge(ZtZ, ZtY_g, compute_device).t().to(
                dtype=L_c.mlp.gate_proj.weight.dtype, 
                device=L_c.mlp.gate_proj.weight.device
            )
            L_c.mlp.up_proj.weight.data = solve_ridge(ZtZ, ZtY_u, compute_device).t().to(
                dtype=L_c.mlp.up_proj.weight.dtype, 
                device=L_c.mlp.up_proj.weight.device
            )
            probe_current_state("[3] 已校准: KQV + O + Gate/Up")

        if TUNE_CONFIG["tune_ffn_o"]:
            target_ffn_o = [(Op + Oc) * damp for Op, Oc in zip(orig_targets['Op_ffn'], orig_targets['Oc_ffn'])]
            collect_Z_and_solve(L_c.mlp.down_proj, target_ffn_o, "校准 FFN down_proj")
            probe_current_state("[4] 已校准: KQV + O + Gate/Up + Down (全鏈路完成)", run_diag=True)

        # === 物理切除 ===
        logging.info(f"🔪 執行物理切除 Layer {prune_idx}")
        modelhander.remove_layers([prune_idx])
        gc.collect(); torch.cuda.empty_cache()

    logging.info(f"✅ FDD 遞進式全鏈路補償算法結束。")
    save_path = getattr(args, 'save_path', f"fdd_progressive_{args.target_layers}L")
    modelhander.save(path=save_path)