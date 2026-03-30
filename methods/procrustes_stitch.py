import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from collections import OrderedDict

from utils.data_utils import get_trainloaders
from utils.eval_utils import load_and_eval_ppl

@torch.enable_grad()
def main_func(args, modelhander):
    logging.info("🚀 启动 MAPI 算法：流形感知迭代剪枝 + 幻影偏置注入...")
    
    # 1. 初始化数据
    dataloader = get_trainloaders(
        args.calibration_dataset,
        tokenizer=modelhander.tokenizer,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=modelhander.model.seqlen
    )

    device = modelhander.model.device
    testenc = dataloader.input_ids
    seqlen = modelhander.model.seqlen
    nsamples = testenc.numel() // seqlen
    
    # 记录原始层号与偏置存储器
    original_indices = list(range(modelhander.model.config.num_hidden_layers))
    phantom_biases = {} # 存储原层号对应的偏置向量
    
    iteration = 0
    total_to_prune = modelhander.model.config.num_hidden_layers - args.target_layers

    # 2. 迭代手术循环
    while modelhander.model.config.num_hidden_layers > args.target_layers:
        iteration += 1
        current_layer_num = modelhander.model.config.num_hidden_layers
        logging.info(f"\n{'='*60}\n🔄 轮次 {iteration}/{total_to_prune} | 当前层数: {current_layer_num}")
        
        layer_grads = {}
        def get_grad_hook(name):
            def hook(module, grad_input, grad_output):
                if name not in layer_grads:
                    layer_grads[name] = []
                # 捕获输出梯度并转为 CPU 上的 float32
                layer_grads[name].append(grad_output[0].detach().cpu().float())
            return hook

        hooks = []
        for idx, layer in enumerate(modelhander.model.model.layers):
            hooks.append(layer.register_full_backward_hook(get_grad_hook(idx)))

        modelhander.model.train() 
        activations = [[] for _ in range(current_layer_num + 1)]

        for i in tqdm(range(nsamples), desc=f"Iter {iteration} 梯度捕获"):
            inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(device)
            modelhander.model.zero_grad()
            
            # 必须关闭 cache 以免 layer_idx 越界
            outputs = modelhander.model(inputs, labels=inputs, output_hidden_states=True, use_cache=False)
            
            for idx, h in enumerate(outputs.hidden_states):
                activations[idx].append(h.detach().cpu().float())
                
            outputs.loss.backward()

        for h in hooks: h.remove()
        modelhander.model.eval()
        torch.cuda.empty_cache()

        # 维度对齐处理
        hidden_dim = modelhander.model.config.hidden_size
        full_activations = [torch.cat(act, dim=0).view(-1, hidden_dim) for act in activations]

        # 3. 核心决策：计算流形感知得分
        scores = {}
        for l_idx in range(1, current_layer_num - 1): # 保护首尾
            X_in = full_activations[l_idx]
            X_out = full_activations[l_idx + 1]
            
            # 聚合多 batch 梯度并展平
            G = torch.cat(layer_grads[l_idx][::-1], dim=0).view(-1, hidden_dim)
            
            # 修复维度报错的核心：确保 delta_X [N, D] 与 G [N, D] 逐元素乘法
            delta_X = X_out - X_in
            base_taylor = torch.mean(torch.abs(delta_X * G)).item()
            
            # 流形曲率惩罚 (1 - CosSim)
            cos_sim = F.cosine_similarity(X_in, X_out, dim=-1).mean().item()
            curvature = max(1.0 - cos_sim, 1e-6)
            
            scores[l_idx] = base_taylor * curvature

        # 挑选牺牲者
        prune_idx = sorted(scores.items(), key=lambda x: x[1])[0][0]
        orig_idx = original_indices[prune_idx]
        
        # 4. 执行幻影偏置注入：计算均值偏移
        # 即使该层被删，我们也要把它的均值贡献留给下一层
        bias_vector = (full_activations[prune_idx + 1] - full_activations[prune_idx]).mean(dim=0).to(device)
        
        # 如果下一层已经是之前被注入过的，我们需要累加偏置（形成偏置链）
        target_orig_idx = original_indices[prune_idx + 1]
        if target_orig_idx in phantom_biases:
            phantom_biases[target_orig_idx] += bias_vector
        else:
            phantom_biases[target_orig_idx] = bias_vector

        logging.info(f"🎯 选定 Layer {prune_idx} (原 L{orig_idx}) | 注入偏置模长: {torch.norm(bias_vector).item():.4f}")

        # 物理删除
        modelhander.remove_layers([prune_idx])
        del original_indices[prune_idx]
        torch.cuda.empty_cache()

    # 5. 部署幻影偏置 Hook 链
    logging.info("💉 正在为剩余层部署幻影偏置注入钩子...")
    
    # 获取模型的目标 dtype (通常是 layer 内部权重的 dtype)
    # 取第一层的 input_layernorm 权重类型作为参考，或者直接用 model.dtype
    target_dtype = modelhander.model.model.layers[0].input_layernorm.weight.dtype

    for current_i, orig_i in enumerate(original_indices):
        if orig_i in phantom_biases:
            layer_module = modelhander.model.model.layers[current_i]
            
            # 【关键修复】将偏置向量转换为与模型一致的 dtype (如 bfloat16)
            bias_vector = phantom_biases[orig_i].to(dtype=target_dtype)
            
            # 使用闭包固化当前的 bias 向量
            def create_hook(b_vec):
                # 确保加法不会导致类型提升，或者在加之前再次确保类型一致
                return lambda m, args: (args[0] + b_vec,)
            
            # 挂载到 input_layernorm 之前的 pre_hook
            layer_module.register_forward_pre_hook(create_hook(bias_vector))

    # 6. 最终评测
    if not args.continue_saving:
        save_path = os.path.join(args.save_path, args.save_name + f"_mapi_final")
        modelhander.save(path=save_path)
        for da in args.ppl_data:
            logging.info(f"正在评测数据集: {da}")
            modelhander.model = modelhander.model.to("cuda:0")
            ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
            logging.info(f"📊 {da} PPL: {ppl}")