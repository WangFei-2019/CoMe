import os
import gc
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging

from utils.data_utils import get_trainloaders
from utils.eval_utils import load_and_eval_ppl

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
        
        logging.info("📊 正在提取 FDD 評估基準線 (Golden Baseline)...")
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
                cos_sim = F.cosine_similarity(
                    self.golden_hiddens[i].float(), 
                    outputs.hidden_states[-1].cpu().float(), 
                    dim=-1
                ).mean().item()
                cos_list.append(cos_sim)
        return np.mean(jsd_list), np.mean(cos_list)

@torch.enable_grad()
def main_func(args, modelhander):
    logging.info("🚀 啟動 FDD-Ridge 剪枝算法 (包含細粒度 PPL 與探針評估)...")
    
    dataloader = get_trainloaders(
        args.calibration_dataset, tokenizer=modelhander.tokenizer,
        nsamples=args.nsamples, seed=args.seed, seqlen=modelhander.model.seqlen
    )

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
        
        # === Phase 1: 敏感度掃描 ===
        taylor_scores = {l: 0.0 for l in range(1, current_layer_num - 1)}
        cos_sim_sums = {l: 0.0 for l in range(1, current_layer_num - 1)}
        total_tokens = 0
        
        modelhander.model.eval()
        for param in modelhander.model.parameters(): param.requires_grad = False
            
        grad_dict = {}
        def get_grad_hook(l_idx):
            return lambda m, gi, go: grad_dict.update({l_idx: go[0].detach()}) if go[0] is not None else None
            
        bw_hooks = [modelhander.model.model.layers[l].register_full_backward_hook(get_grad_hook(l)) for l in range(1, current_layer_num - 1)]
            
        for i in tqdm(range(nsamples), desc=f"探針掃描"):
            inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(compute_device)
            inputs_embeds = modelhander.model.get_input_embeddings()(inputs).detach().clone().requires_grad_(True)
            outputs = modelhander.model(inputs_embeds=inputs_embeds, labels=inputs, output_hidden_states=True, use_cache=False)
            outputs.loss.backward()
            
            for l_idx in range(1, current_layer_num - 1):
                G = grad_dict[l_idx]
                target_dev = G.device 
                X_in = outputs.hidden_states[l_idx].detach().to(target_dev)
                X_out = outputs.hidden_states[l_idx+1].detach().to(target_dev)
                
                delta_X = X_out - X_in
                taylor_scores[l_idx] += torch.sum(torch.abs(delta_X * G)).item()
                cos_sim_sums[l_idx] += F.cosine_similarity(X_in.float(), X_out.float(), dim=-1).sum().item()
                
            total_tokens += (inputs.shape[0] * inputs.shape[1])
            modelhander.model.zero_grad()
            grad_dict.clear() 
            del outputs, inputs_embeds; torch.cuda.empty_cache()

        for h in bw_hooks: h.remove()

        scores = {l: (taylor_scores[l]/total_tokens) * max(1.0 - (cos_sim_sums[l]/total_tokens), 1e-6) for l in range(1, current_layer_num - 1)}
        prune_idx = min(scores, key=scores.get)
        comp_idx = prune_idx + 1
        logging.info(f"🎯 鎖定層: Layer {prune_idx} (最低分 {scores[prune_idx]:.6f})，將由 Layer {comp_idx} 吸收。")

        # === 建立模擬評估工具 (升級版：加入 PPL 測試) ===
        layers = modelhander.model.model.layers
        L_p, L_c = layers[prune_idx], layers[comp_idx]
        
        def probe_current_state(stage_name):
            logging.info(f"\n--- 🧪 開始評估: {stage_name} ---")
            # 掛載 Hook 模擬該層已被物理刪除
            bypass_hook = L_p.register_forward_hook(lambda m, i, o: (i[0],) + o[1:] if isinstance(o, tuple) else i[0])
            
            # 1. 快速 FDD 探針
            jsd, cos = fdd_evaluator.evaluate(modelhander.model)
            
            # 2. 全局 PPL 探針
            wiki_ppl = load_and_eval_ppl(modelhander.model, dataset="wiki2", tokenizer=modelhander.tokenizer)
            c4_ppl = load_and_eval_ppl(modelhander.model, dataset="c4", tokenizer=modelhander.tokenizer)
            
            bypass_hook.remove()
            logging.info(f"➤ [{stage_name}] 診斷結果: JSD: {jsd:.6f} | Cos: {cos:.4f} | Wiki2: {wiki_ppl:.2f} | C4: {c4_ppl:.2f}\n")

        # 🚨 [監控 1]
        probe_current_state("1. 原始直接剪除 (未補償)")

        # === Phase 2: 嶺迴歸補償 ===
        hooks = SurgeryHooks()
        num_calib = min(nsamples, 32)
        damping_factor = 0.8 

        Y_attn_cache, Y_ffn_cache = [], []
        for i in range(num_calib):
            inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(compute_device)
            hooks.handles.extend([
                L_p.self_attn.o_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Op_attn': o.detach().cpu()})),
                L_c.self_attn.o_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Oc_attn': o.detach().cpu()})),
                L_p.mlp.down_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Op_ffn': o.detach().cpu()})),
                L_c.mlp.down_proj.register_forward_hook(lambda m, i, o: hooks.data.update({'Oc_ffn': o.detach().cpu()}))
            ])
            with torch.no_grad(): modelhander.model(inputs, use_cache=False)
            hooks.remove()
            Y_attn_cache.append((hooks.data['Op_attn'] + hooks.data['Oc_attn']) * damping_factor)
            Y_ffn_cache.append((hooks.data['Op_ffn'] + hooks.data['Oc_ffn']) * damping_factor)
            hooks.clear()

        # [A] Attention 補償
        D_attn = L_c.self_attn.o_proj.weight.shape[1]
        D_out_attn = L_c.self_attn.o_proj.weight.shape[0]
        ZtZ, ZtY = torch.zeros((D_attn, D_attn), device=compute_device), torch.zeros((D_attn, D_out_attn), device=compute_device)
        for i in range(num_calib):
            inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(compute_device)
            hooks.handles.extend([
                L_p.register_forward_hook(lambda m, i, o: (i[0],) + o[1:] if isinstance(o, tuple) else i[0]),
                L_c.self_attn.o_proj.register_forward_hook(lambda m, inp, o: hooks.data.update({'Z': inp[0].detach().cpu()}))
            ])
            with torch.no_grad(): modelhander.model(inputs, use_cache=False)
            hooks.remove()
            Z, Y = hooks.data['Z'].view(-1, D_attn).float().to(compute_device), Y_attn_cache[i].view(-1, D_out_attn).float().to(compute_device)
            ZtZ += Z.t() @ Z; ZtY += Z.t() @ Y
            hooks.clear()
            
        W_comp = torch.linalg.solve(ZtZ + (0.01 * torch.trace(ZtZ)/D_attn) * 
        torch.eye(D_attn, device=compute_device), ZtY)

        # # --- 🚀 新增：Attention 方差保持 (Variance Preservation) ---
        # with torch.no_grad():
        #     # Z 和 Y 此時是最後一個 Batch 的殘留數據，剛好可以用來做方差探針
        #     Y_pred = Z @ W_comp
        #     std_target = Y.std()
        #     std_pred = Y_pred.std()
            
        #     # 計算放大係數 (如果預測方差縮水了，就把它等比例放大)
        #     scale_factor = std_target / (std_pred + 1e-6)
            
        #     # 限制放大倍率，防止因為極端情況導致數值爆炸 (1.0 代表不縮水)
        #     scale_factor = torch.clamp(scale_factor, 1.0, 1.5) 
            
        #     W_comp = W_comp * scale_factor
        #     logging.info(f"      [方差校正] Attn 放大係數: {scale_factor.item():.4f}")
        # # --------------------------------------------------------

        L_c.self_attn.o_proj.weight.data = W_comp.t().to(dtype=L_c.self_attn.o_proj.weight.dtype, device=L_c.self_attn.o_proj.weight.device)

        # 🚨 [監控 2]
        probe_current_state("2. 僅 Attention 被補償")

        # [B] FFN 補償
        D_ffn, D_out_ffn = L_c.mlp.down_proj.weight.shape[1], L_c.mlp.down_proj.weight.shape[0]
        ZtZ, ZtY = torch.zeros((D_ffn, D_ffn), device=compute_device), torch.zeros((D_ffn, D_out_ffn), device=compute_device)
        for i in range(num_calib):
            inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(compute_device)
            hooks.handles.extend([
                L_p.register_forward_hook(lambda m, i, o: (i[0],) + o[1:] if isinstance(o, tuple) else i[0]),
                L_c.mlp.down_proj.register_forward_hook(lambda m, inp, o: hooks.data.update({'Z': inp[0].detach().cpu()}))
            ])
            with torch.no_grad(): modelhander.model(inputs, use_cache=False)
            hooks.remove()
            Z, Y = hooks.data['Z'].view(-1, D_ffn).float().to(compute_device), Y_ffn_cache[i].view(-1, D_out_ffn).float().to(compute_device)
            ZtZ += Z.t() @ Z; ZtY += Z.t() @ Y
            hooks.clear()
            
        W_comp = torch.linalg.solve(ZtZ + (0.01 * torch.trace(ZtZ)/D_ffn) * torch.eye(D_ffn, device=compute_device), ZtY)
        
        # # --- 🚀 新增：FFN 方差保持 (Variance Preservation) ---
        # with torch.no_grad():
        #     Y_pred = Z @ W_comp
        #     std_target = Y.std()
        #     std_pred = Y_pred.std()
            
        #     scale_factor = std_target / (std_pred + 1e-6)
        #     scale_factor = torch.clamp(scale_factor, 1.0, 1.5) 
            
        #     W_comp = W_comp * scale_factor
        #     logging.info(f"      [方差校正] FFN 放大係數: {scale_factor.item():.4f}")
        # # --------------------------------------------------------

        L_c.mlp.down_proj.weight.data = W_comp.t().to(dtype=L_c.mlp.down_proj.weight.dtype, device=L_c.mlp.down_proj.weight.device)

        # 🚨 [監控 3]
        probe_current_state("3. Attn + FFN 雙重補償")

        # === Phase 3: 物理切除 ===
        logging.info(f"🔪 執行物理切除 Layer {prune_idx}")
        modelhander.remove_layers([prune_idx])
        gc.collect(); torch.cuda.empty_cache()

    logging.info(f"✅ FDD-Ridge 剪枝結束。")
    save_path = args.save_path if args.save_path else f"fdd_ridge_pruned_{args.target_layers}L"
    modelhander.save(path=save_path)
    
    # 最後不需要再測一次了，因為最後一層剪完的 [監控 3] 已經測過了