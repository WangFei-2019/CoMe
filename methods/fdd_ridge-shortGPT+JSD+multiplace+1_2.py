import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging

from utils.data_utils import get_trainloaders
from utils.eval_utils import load_and_eval_ppl

# ==========================================
# 🎛️ 递进式校准控制器
# ==========================================
TUNE_CONFIG = {
    "tune_kqv": True,      # 是否校准 L+1 的 q_proj, k_proj, v_proj
    "tune_attn_o": True,    # 是否校准 L+1 的 o_proj
    "tune_ffn_in": True,   # 是否校准 L+1 的 gate_proj, up_proj
    "tune_ffn_o": True,     # 是否校准 L+1 的 down_proj
    "damping_factor": 1.0   # 目标阻尼系数 (防方差爆炸)
}

# ==========================================
# 核心辅助函数 (统一放到顶部)
# ==========================================
def compute_token_difficulty_by_prob(golden_logits_batch, input_ids_batch):
    """根据原模型的预测置信度来分配权重"""
    probs = F.softmax(golden_logits_batch.float(), dim=-1)
    batch_size, seq_len = input_ids_batch.shape
    target_probs = torch.zeros((batch_size, seq_len), device=golden_logits_batch.device)
    
    for b in range(batch_size):
        for s in range(seq_len - 1):
            target_token = input_ids_batch[b, s+1]
            target_probs[b, s] = probs[b, s, target_token]
        target_probs[b, seq_len-1] = 1.0 
        
    weights = 1.0 / (target_probs + 1e-5) 
    weights = torch.clamp(weights, min=0.5, max=5.0)
    return weights.view(-1, 1)

def diagnose_ppl_jsd_conflict(golden_logits, uncomp_logits, comp_logits, golden_hiddens, comp_hiddens, input_ids, tokenizer, save_dir, layer_name):
    """微观诊断探针：打印在线分析，并将全量 Token 数据保存用于离线交叉分析"""
    shift_labels = input_ids[..., 1:].contiguous().view(-1)
    
    g_logits = golden_logits[..., :-1, :].contiguous().view(-1, golden_logits.size(-1))
    u_logits = uncomp_logits[..., :-1, :].contiguous().view(-1, uncomp_logits.size(-1))
    c_logits = comp_logits[..., :-1, :].contiguous().view(-1, comp_logits.size(-1))
    
    ce_gold = F.cross_entropy(g_logits, shift_labels, reduction='none')
    ce_uncomp = F.cross_entropy(u_logits, shift_labels, reduction='none')
    ce_comp = F.cross_entropy(c_logits, shift_labels, reduction='none')
    
    ce_diff = ce_comp - ce_gold
    
    h_gold = golden_hiddens[..., :-1, :].contiguous().view(-1, golden_hiddens.size(-1))
    h_comp = comp_hiddens[..., :-1, :].contiguous().view(-1, comp_hiddens.size(-1))
    norm_ratio = (h_comp.norm(dim=-1) / h_gold.norm(dim=-1)).mean().item()
    
    # 打印在线诊断报告
    top_k = 15
    worst_indices = torch.topk(ce_diff, top_k).indices
    print("\n" + "="*80)
    print(f"🔍 [深度诊断] JSD 与 PPL 背离现象分析报告 ({layer_name})")
    print("="*80)
    print(f"📉 全局特征模长收缩比 (Comp Norm / Gold Norm): {norm_ratio:.4f}")
    if norm_ratio < 0.9: print("   ⚠️ 警告：特征模长严重收缩！会导致 Logits 变平缓拉高 PPL！")
    elif norm_ratio > 1.1: print("   ⚠️ 警告：特征模长爆炸！会触发 Logits 极端尖锐化！")
        
    print(f"\n🚨 退化最严重的 Top-{top_k} 个 Token:")
    print(f"{'Target Token':<15} | {'Gold Pred (Prob)':<20} | {'Comp Pred (Prob)':<20} | {'Δ CE Loss':<10}")
    print("-" * 80)
    for idx in worst_indices:
        target_token_id = shift_labels[idx].item()
        target_word = tokenizer.decode([target_token_id]).replace('\n', '\\n')
        gold_probs = F.softmax(g_logits[idx], dim=-1)
        comp_probs = F.softmax(c_logits[idx], dim=-1)
        gold_prob_val = gold_probs[target_token_id].item()
        comp_prob_val = comp_probs[target_token_id].item()
        gold_top1_word = tokenizer.decode([torch.argmax(gold_probs).item()]).replace('\n', '\\n')
        comp_top1_word = tokenizer.decode([torch.argmax(comp_probs).item()]).replace('\n', '\\n')
        print(f"[{target_word:<13}] | {gold_top1_word:<10} ({gold_prob_val:.2%}) | {comp_top1_word:<10} ({comp_prob_val:.2%}) | +{ce_diff[idx].item():.3f}")
    print("="*80 + "\n")

    # 💾 保存离线可视化数据
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        prob_gold = F.softmax(g_logits, dim=-1).gather(1, shift_labels.unsqueeze(1)).squeeze()
        prob_uncomp = F.softmax(u_logits, dim=-1).gather(1, shift_labels.unsqueeze(1)).squeeze()
        prob_comp = F.softmax(c_logits, dim=-1).gather(1, shift_labels.unsqueeze(1)).squeeze()
        export_data = {
            "tokens": shift_labels.cpu(),
            "ce_gold": ce_gold.cpu(), "ce_uncomp": ce_uncomp.cpu(), "ce_comp": ce_comp.cpu(),
            "prob_gold": prob_gold.cpu(), "prob_uncomp": prob_uncomp.cpu(), "prob_comp": prob_comp.cpu()
        }
        save_path = os.path.join(save_dir, f"token_diag_{layer_name}.pt")
        torch.save(export_data, save_path)
        logging.info(f"💾 离线可视化数据已保存至: {save_path}")

class SurgeryHooks:
    def __init__(self): self.handles = []; self.data = {}
    def clear(self): self.data.clear()
    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

def compute_jsd(logits_p, logits_q):
    p = F.softmax(logits_p.float(), dim=-1)
    q = F.softmax(logits_q.float(), dim=-1)
    m = 0.5 * (p + q)
    return (0.5 * (F.kl_div(m.log(), torch.clamp(p, min=1e-8), reduction='none').sum(dim=-1) + 
                   F.kl_div(m.log(), torch.clamp(q, min=1e-8), reduction='none').sum(dim=-1))).mean().item()

class FDDEvaluator:
    def __init__(self, model, dataloader, num_batches=4):
        self.valid_inputs = []
        self.golden_logits = []
        self.golden_hiddens = []
        self.device = model.device
        model.eval()
        with torch.no_grad():
            for i in range(num_batches):
                inputs = dataloader.input_ids[:, (i * model.seqlen):((i+1) * model.seqlen)].to(self.device)
                self.valid_inputs.append(inputs)
                outputs = model(inputs, output_hidden_states=True, use_cache=False)
                self.golden_logits.append(outputs.logits.cpu())
                self.golden_hiddens.append(outputs.hidden_states[-1].cpu())

    def evaluate(self, model):
        model.eval()
        jsd_list, cos_list = [], []
        with torch.no_grad():
            for i, inputs in enumerate(self.valid_inputs):
                outputs = model(inputs, output_hidden_states=True, use_cache=False)
                jsd_list.append(compute_jsd(self.golden_logits[i], outputs.logits.cpu()))
                cos_list.append(F.cosine_similarity(self.golden_hiddens[i].float(), outputs.hidden_states[-1].cpu().float(), dim=-1).mean().item())
        return np.mean(jsd_list), np.mean(cos_list)
    
    def run_diagnosis(self, model, tokenizer, L_p, save_dir, layer_name):
        """同时获取补偿前后状态，触发探针并保存数据"""
        model.eval()
        with torch.no_grad():
            inputs = self.valid_inputs[0]
            gold_logits = self.golden_logits[0].to(self.device)
            gold_hiddens = self.golden_hiddens[0].to(self.device)
            
            # 1. 补偿后 (当前模型状态)
            outputs_comp = model(inputs, output_hidden_states=True, use_cache=False)
            comp_logits = outputs_comp.logits
            comp_hiddens = outputs_comp.hidden_states[-1]
            
            # 2. 未补偿 (挂上跳过钩子模拟直接剪枝)
            bypass_hook = L_p.register_forward_hook(lambda m, i, o: (i[0],) + o[1:] if isinstance(o, tuple) else i[0])
            outputs_uncomp = model(inputs, output_hidden_states=True, use_cache=False)
            uncomp_logits = outputs_uncomp.logits
            bypass_hook.remove()
            
            diagnose_ppl_jsd_conflict(
                gold_logits, uncomp_logits, comp_logits, 
                gold_hiddens, comp_hiddens, 
                inputs, tokenizer, save_dir, layer_name
            )

def solve_ridge(Z, Y, device, reg_ratio=0.01):
    D = Z.shape[0]
    norm = torch.trace(Z) / D
    lambda_reg = reg_ratio * norm if norm > 0 else 1e-4
    return torch.linalg.solve(Z + lambda_reg * torch.eye(D, device=device), Y)

@torch.inference_mode()
def main_func(args, modelhander):
    dataloader = get_trainloaders(args.calibration_dataset, tokenizer=modelhander.tokenizer, nsamples=args.nsamples, seed=args.seed, seqlen=modelhander.model.seqlen)
    compute_device = modelhander.model.device
    testenc = dataloader.input_ids
    seqlen = modelhander.model.seqlen
    nsamples = testenc.numel() // seqlen
    
    fdd_evaluator = FDDEvaluator(modelhander.model, dataloader, num_batches=4)
    total_to_prune = modelhander.model.config.num_hidden_layers - args.target_layers

    # 预先获取存数据的目录
    save_dir = getattr(args, 'save_path', f"fdd_progressive_{args.target_layers}L")

    iteration = 0
    while modelhander.model.config.num_hidden_layers > args.target_layers:
        iteration += 1
        current_layer_num = modelhander.model.config.num_hidden_layers
        logging.info(f"\n{'='*70}\n🔄 輪次 {iteration}/{total_to_prune} | 當前層數: {current_layer_num}")
        
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
        
        def probe_current_state(stage_name):
            logging.info(f"--- 🧪 探針: {stage_name} ---")
            bypass_hook = L_p.register_forward_hook(lambda m, i, o: (i[0],) + o[1:] if isinstance(o, tuple) else i[0])
            jsd, cos = fdd_evaluator.evaluate(modelhander.model)
            w_ppl = load_and_eval_ppl(modelhander.model, dataset="wiki2", tokenizer=modelhander.tokenizer)
            bypass_hook.remove()
            logging.info(f"➤ JSD: {jsd:.6f} | Cos: {cos:.4f} | Wiki2: {w_ppl:.2f} \n")
                
            bypass_hook.remove()
            logging.info(f"➤ JSD: {jsd:.6f} | Cos: {cos:.4f} | Wiki2: {w_ppl:.2f}\n")

        probe_current_state("[0] 直接剪除 (無補償)")

        hooks = SurgeryHooks()
        num_calib = min(nsamples, 32)
        damp = TUNE_CONFIG["damping_factor"]

        # Step 1: 提取原模型 Targets
        orig_targets = {'Q': [], 'K': [], 'V': [], 'Op_attn': [], 'Oc_attn': [], 'Gate': [], 'Up': [], 'Op_ffn': [], 'Oc_ffn': [], 'Token_Weights': []}
        for i in tqdm(range(num_calib), desc="提取原模型目標特徵"):
            inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(compute_device)
            hooks.handles.extend([
                L_c.self_attn.q_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Q': o.detach().cpu()})),
                L_c.self_attn.k_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'K': o.detach().cpu()})),
                L_c.self_attn.v_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'V': o.detach().cpu()})),
                L_p.self_attn.o_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Op_attn': o.detach().cpu()})),
                L_c.self_attn.o_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Oc_attn': o.detach().cpu()})),
                L_c.mlp.gate_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Gate': o.detach().cpu()})),
                L_c.mlp.up_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Up': o.detach().cpu()})),
                L_p.mlp.down_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Op_ffn': o.detach().cpu()})),
                L_c.mlp.down_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Oc_ffn': o.detach().cpu()}))
            ])
            outputs = modelhander.model(inputs, use_cache=False)
            hooks.remove()
            for key in orig_targets.keys():
                if key != 'Token_Weights': orig_targets[key].append(hooks.data[key])
            
            # 缓存 Method 2 权重
            batch_weights = compute_token_difficulty_by_prob(outputs.logits, inputs)
            orig_targets['Token_Weights'].append(batch_weights.detach().cpu())
            hooks.clear()

        # 方法一：L2 Norm 加权
        def collect_Z_and_solve_method_1(proj_layer, target_list, desc):
            D_in, D_out = proj_layer.weight.shape[1], proj_layer.weight.shape[0]
            ZtZ, ZtY = torch.zeros((D_in, D_in), device=compute_device), torch.zeros((D_in, D_out), device=compute_device)
            for i in tqdm(range(num_calib), desc=desc):
                inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(compute_device)
                hooks.handles.extend([
                    L_p.register_forward_hook(lambda m, i, o: (i[0],) + o[1:] if isinstance(o, tuple) else i[0]),
                    proj_layer.register_forward_hook(lambda m, inp, o: hooks.data.update({'Z': inp[0].detach().cpu()}))
                ])
                modelhander.model(inputs, use_cache=False)
                hooks.remove()
                Z = hooks.data['Z'].view(-1, D_in).float().to(compute_device)
                Y = target_list[i].view(-1, D_out).float().to(compute_device)
                
                # Norm 加权
                token_norms = Y.norm(dim=-1, keepdim=True)
                weights = torch.clamp((token_norms / token_norms.mean()) ** 2, min=0.5, max=5.0)
                ZtZ += Z.t() @ (Z * weights)
                ZtY += Z.t() @ (Y * weights)
                hooks.clear()
            proj_layer.weight.data = solve_ridge(ZtZ, ZtY, compute_device).t().to(dtype=proj_layer.weight.dtype, device=proj_layer.weight.device)
        # 方法二：基于 Golden Model 预测概率的加权 (使用 Step 1 缓存的权重)
        def collect_Z_and_solve_method_2(proj_layer, target_list, desc):
            D_in, D_out = proj_layer.weight.shape[1], proj_layer.weight.shape[0]
            ZtZ, ZtY = torch.zeros((D_in, D_in), device=compute_device), torch.zeros((D_in, D_out), device=compute_device)
            
            for i in tqdm(range(num_calib), desc=desc):
                inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(compute_device)
                hooks.handles.extend([
                    L_p.register_forward_hook(lambda m, i, o: (i[0],) + o[1:] if isinstance(o, tuple) else i[0]),
                    proj_layer.register_forward_hook(lambda m, inp, o: hooks.data.update({'Z': inp[0].detach().cpu()}))
                ])
                modelhander.model(inputs, use_cache=False)
                hooks.remove()
                
                Z = hooks.data['Z'].view(-1, D_in).float().to(compute_device)
                Y = target_list[i].view(-1, D_out).float().to(compute_device)
                
                # 🌟 从缓存中提取真正的 Token 置信度权重
                weights = orig_targets['Token_Weights'][i].to(compute_device)
                
                Z_weighted = Z * weights
                ZtZ += Z.t() @ Z_weighted
                ZtY += Z.t() @ (Y * weights)
                hooks.clear()
            
            new_W = solve_ridge(ZtZ, ZtY, compute_device)
            proj_layer.weight.data = new_W.t().to(dtype=proj_layer.weight.dtype, device=proj_layer.weight.device)

        collect_Z_and_solve = collect_Z_and_solve_method_1

        # Step 2: 递进校准
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
            probe_current_state("[4] 已校准: KQV + O + Gate/Up + Down (全鏈路完成)")

        # 🌟 【触发离线保存逻辑】：只在这里存一次即可
        fdd_evaluator.run_diagnosis(modelhander.model, modelhander.tokenizer, L_p, save_dir, f"Layer{prune_idx}")

        # === 物理切除 ===
        logging.info(f"🔪 執行物理切除 Layer {prune_idx}")
        modelhander.remove_layers([prune_idx])
        gc.collect(); torch.cuda.empty_cache()

    modelhander.save(path=save_dir)