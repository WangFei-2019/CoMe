# import os
# import json
# import torch
# from tqdm import tqdm
# import logging
# import torch.nn.functional as F

# from utils.data_utils import get_trainloaders
# from utils.eval_utils import load_and_eval_ppl

# @torch.enable_grad()
# def main_func(args, modelhander):
#     # 1. 获取校准数据
#     dataloader = get_trainloaders(
#         args.calibration_dataset,
#         tokenizer=modelhander.tokenizer,
#         nsamples=args.nsamples,
#         seed=args.seed,
#         seqlen=modelhander.model.seqlen
#     )

#     device = modelhander.model.device
#     testenc = dataloader.input_ids
#     seqlen = modelhander.model.seqlen
#     nsamples = testenc.numel() // seqlen
    
#     layer_grads = {}
    
#     # 钩子：在 CPU 上累加梯度，防止多卡 OOM
#     def get_grad_hook(name):
#         def hook(module, grad_input, grad_output):
#             if name not in layer_grads:
#                 layer_grads[name] = []
#             layer_grads[name].append(grad_output[0].detach().cpu())
#         return hook

#     hooks = []
#     for idx, layer in enumerate(modelhander.model.model.layers):
#         hooks.append(layer.register_full_backward_hook(get_grad_hook(idx)))

#     logging.info("🚀 [TGRC] 正在执行多卡前向与反向传播，捕获全局梯度...")
#     modelhander.model.train() 
    
#     layer_num = modelhander.model.config.num_hidden_layers
#     activations = [[] for _ in range(layer_num + 1)]

#     batch_size = 1
#     for i in tqdm(range(0, nsamples, batch_size), desc="Global Gradient Capture"):
#         j = min(i+batch_size, nsamples)
#         inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
#         inputs = inputs.reshape(j-i, seqlen)

#         modelhander.model.zero_grad()
#         outputs = modelhander.model(inputs, labels=inputs, output_hidden_states=True)
#         loss = outputs.loss
        
#         hidden_states = outputs.hidden_states
#         for idx in range(len(hidden_states)):
#             activations[idx].append(hidden_states[idx].view(-1, hidden_states[idx].shape[-1]).detach().cpu())
            
#         loss.backward()

#     for h in hooks:
#         h.remove()
#     modelhander.model.eval()
#     modelhander.model.zero_grad()
#     torch.cuda.empty_cache()

#     activations = [torch.cat(act, dim=0) for act in activations]

#     logging.info("📊 正在计算泰勒全局敏感度 (Taylor Global Sensitivity) ...")
#     taylor_scores = {}
#     for l_idx in range(1, layer_num - 1): 
#         X_in = activations[l_idx]       
#         X_out = activations[l_idx + 1]  
        
#         G = torch.cat(layer_grads[l_idx][::-1], dim=0).view(-1, X_in.shape[-1])
        
#         delta_X = X_in - X_out
#         score = torch.mean(torch.abs(delta_X * G)).item()
#         taylor_scores[l_idx] = score
        
#         del layer_grads[l_idx]
#         logging.info(f"Layer {l_idx:02d} | 泰勒全局敏感度得分: {score:.8f}")

#     sorted_layers = sorted(taylor_scores.items(), key=lambda x: x[1])
#     layers_to_drop = modelhander.model.config.num_hidden_layers - args.target_layers
#     prune_indices = [x[0] for x in sorted_layers[:layers_to_drop]]
#     prune_indices.sort(reverse=True)
    
#     logging.info(f"\n🎯 泰勒指标决定剪除以下 {layers_to_drop} 层 (深到浅): {prune_indices}")

#     # ==========================
#     # 物理删除层
#     # ==========================
#     for prune_idx in prune_indices:
#         modelhander.remove_layers([prune_idx])
#     logging.info(f"✅ 物理剪枝完成，当前层数: {modelhander.model.config.num_hidden_layers}")

#     # ==========================
#     # 核心创新：多卡自适应脊回归补偿 (Ridge Regression)
#     # ==========================
#     surviving_indices = [i for i in range(layer_num) if i not in prune_indices]
    
#     boundaries = []
#     for i in range(1, len(surviving_indices)):
#         if surviving_indices[i] > surviving_indices[i-1] + 1:
#             boundaries.append(surviving_indices[i])

#     tokens_per_sample = 128 
    
#     for target_orig_idx in boundaries:
#         curr_idx = surviving_indices.index(target_orig_idx)
#         layer_to_fix = modelhander.model.model.layers[curr_idx]
        
#         # 获取当前层所在的物理 GPU 设备，后续所有计算都将迁移到该卡上！
#         layer_device = layer_to_fix.mlp.down_proj.weight.device
        
#         logging.info(f"🔧 正在执行补偿: 修正当前网络 Layer {curr_idx} (对应原 Layer {target_orig_idx}) [计算设备: {layer_device}]...")
        
#         H_sub_list, Y_act_sub_list, Y_tgt_sub_list = [], [], []
#         current_H, current_Y = [], []
        
#         def hook_H(m, args, out):
#             current_H.append(args[0].detach())
#         def hook_Y(m, args, out):
#             current_Y.append(out[0].detach())
            
#         h1 = layer_to_fix.mlp.down_proj.register_forward_hook(hook_H)
#         h2 = layer_to_fix.register_forward_hook(hook_Y)
        
#         for i in tqdm(range(nsamples), desc=f"Aligning Layer {curr_idx}"):
#             # 保证输入在主设备上以启动前向传播，HuggingFace 的 device_map 会自动处理跨卡传递
#             inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(device)
#             current_H.clear()
#             current_Y.clear()
            
#             with torch.no_grad():
#                 modelhander.model(inputs) 
                
#             H_batch = current_H[0][0]     
#             Y_act_batch = current_Y[0][0] 
            
#             # Target 特征原本在 CPU，截取对应 batch
#             Y_tgt_batch = activations[target_orig_idx + 1][i*seqlen : (i+1)*seqlen]
            
#             rand_idx = torch.randperm(seqlen)[:tokens_per_sample]
#             # 全部放回 CPU 列表中缓存，防止循环中爆显存
#             H_sub_list.append(H_batch[rand_idx].cpu())
#             Y_act_sub_list.append(Y_act_batch[rand_idx].cpu())
#             Y_tgt_sub_list.append(Y_tgt_batch[rand_idx].cpu())
            
#         h1.remove()
#         h2.remove()
        
#         # 统一将张量搬运到当前层所在的设备 (layer_device)
#         H_sub = torch.cat(H_sub_list, dim=0).to(torch.float32).to(layer_device)
#         Y_act_sub = torch.cat(Y_act_sub_list, dim=0).to(torch.float32).to(layer_device)
#         Y_tgt_sub = torch.cat(Y_tgt_sub_list, dim=0).to(torch.float32).to(layer_device)
        
#         W_orig = layer_to_fix.mlp.down_proj.weight.data.to(torch.float32)
        
#         Delta_T = Y_tgt_sub - Y_act_sub 
        
#         D = H_sub.shape[1]
#         HTH = torch.matmul(H_sub.T, H_sub)
#         HTT = torch.matmul(H_sub.T, Delta_T)
        
#         damp = 0.05 
#         damp_val = damp * torch.trace(HTH) / D
#         # 必须确保正则化对角矩阵也创建在当前卡的显存上
#         reg_matrix = torch.eye(D, device=layer_device, dtype=torch.float32) * damp_val
        
#         A = HTH + reg_matrix
        
#         # 在当前卡上极速求解
#         dW_T = torch.linalg.solve(A, HTT)
        
#         # 直接就地累加，设备绝对匹配
#         layer_to_fix.mlp.down_proj.weight.data.add_(dW_T.T.to(layer_to_fix.mlp.down_proj.weight.dtype))
#         logging.info(f"✅ Layer {curr_idx} 脊回归残差补偿完成！")

#     # ==========================
#     # 保存与评测
#     # ==========================
#     if not args.continue_saving:
#         save_path = os.path.join(args.save_path, args.save_name + f"_tgrc_{modelhander.model.config.num_hidden_layers}")
#         modelhander.save(path=save_path)
#         for da in args.ppl_data:
#             ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
#             logging.info(f"{da} PPL: {ppl}")


import os
import json
import torch
from tqdm import tqdm
import logging
import torch.nn.functional as F

from utils.data_utils import get_trainloaders
from utils.eval_utils import load_and_eval_ppl

@torch.enable_grad()
def main_func(args, modelhander):
    # 1. 获取校准数据
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
    
    procrustes_error_stats = {}
    iteration = 0
    
    # 动态迭代剪枝，每次只剪 1 层！
    while modelhander.model.config.num_hidden_layers > args.target_layers:
        iteration += 1
        current_layer_num = modelhander.model.config.num_hidden_layers
        logging.info(f"\n{'='*50}")
        logging.info(f"🚀 开始第 {iteration} 轮 i-TGRC 迭代 (当前层数: {current_layer_num} -> 目标: {current_layer_num - 1})")
        logging.info(f"{'='*50}")
        
        layer_grads = {}
        
        # 每次循环重新挂载 Hook
        def get_grad_hook(name):
            def hook(module, grad_input, grad_output):
                if name not in layer_grads:
                    layer_grads[name] = []
                layer_grads[name].append(grad_output[0].detach().cpu())
            return hook

        hooks = []
        for idx, layer in enumerate(modelhander.model.model.layers):
            hooks.append(layer.register_full_backward_hook(get_grad_hook(idx)))

        logging.info("步骤 1/3: 执行多卡前向与反向传播，捕获全局梯度上帝视角...")
        modelhander.model.train() 
        
        activations = [[] for _ in range(current_layer_num + 1)]

        batch_size = 1
        for i in tqdm(range(0, nsamples, batch_size), desc=f"Iter {iteration} - Gradient Capture"):
            j = min(i+batch_size, nsamples)
            inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
            inputs = inputs.reshape(j-i, seqlen)

            modelhander.model.zero_grad()
            outputs = modelhander.model(inputs, labels=inputs, output_hidden_states=True)
            loss = outputs.loss
            
            hidden_states = outputs.hidden_states
            for idx in range(len(hidden_states)):
                activations[idx].append(hidden_states[idx].view(-1, hidden_states[idx].shape[-1]).detach().cpu())
                
            loss.backward()

        # 清理钩子并清空 CUDA 缓存
        for h in hooks:
            h.remove()
        modelhander.model.eval()
        modelhander.model.zero_grad()
        torch.cuda.empty_cache()

        activations = [torch.cat(act, dim=0) for act in activations]

        logging.info("步骤 2/3: 计算泰勒全局敏感度并选出最冗余层...")
        taylor_scores = {}
        for l_idx in range(1, current_layer_num - 1): 
            X_in = activations[l_idx]       
            X_out = activations[l_idx + 1]  
            
            G = torch.cat(layer_grads[l_idx][::-1], dim=0).view(-1, X_in.shape[-1])
            
            delta_X = X_in - X_out
            score = torch.mean(torch.abs(delta_X * G)).item()
            taylor_scores[l_idx] = score
            
            del layer_grads[l_idx]

        sorted_layers = sorted(taylor_scores.items(), key=lambda x: x[1])
        prune_layer_idx = sorted_layers[0][0] # 每次只挑出得分最低的 1 层！
        min_score = sorted_layers[0][1]
        
        logging.info(f"🎯 泰勒指标决定剪除 Layer {prune_layer_idx} (敏感度得分: {min_score:.8f})")

        # 提前提取出被剪枝层原本期望传递给下一层的 Target 特征
        Y_tgt_full = activations[prune_layer_idx + 1].clone()

        # ==========================
        # 物理删除该层
        # ==========================
        modelhander.remove_layers([prune_layer_idx])
        
        # 删除完后，原本在 prune_layer_idx + 1 的层，现在索引变成了 prune_layer_idx
        layer_to_fix_idx = prune_layer_idx
        layer_to_fix = modelhander.model.model.layers[layer_to_fix_idx]
        layer_device = layer_to_fix.mlp.down_proj.weight.device
        
        logging.info(f"步骤 3/3: 物理剪枝完毕。即刻对断点层 Layer {layer_to_fix_idx} 执行闭式解残差补偿 (设备: {layer_device})...")

        # ==========================
        # 即时补偿 (Ridge Regression)
        # ==========================
        H_sub_list, Y_act_sub_list, Y_tgt_sub_list = [], [], []
        current_H, current_Y = [], []
        
        def hook_H(m, args, out):
            current_H.append(args[0].detach())
        def hook_Y(m, args, out):
            current_Y.append(out[0].detach())
            
        h1 = layer_to_fix.mlp.down_proj.register_forward_hook(hook_H)
        h2 = layer_to_fix.register_forward_hook(hook_Y)
        
        tokens_per_sample = 128
        
        for i in tqdm(range(nsamples), desc=f"Iter {iteration} - Aligning Layer {layer_to_fix_idx}"):
            inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(device)
            current_H.clear()
            current_Y.clear()
            
            with torch.no_grad():
                modelhander.model(inputs) 
                
            H_batch = current_H[0][0]     
            Y_act_batch = current_Y[0][0] 
            
            # 从之前缓存的满状态特征中获取 Target
            Y_tgt_batch = Y_tgt_full[i*seqlen : (i+1)*seqlen]
            
            rand_idx = torch.randperm(seqlen)[:tokens_per_sample]
            H_sub_list.append(H_batch[rand_idx].cpu())
            Y_act_sub_list.append(Y_act_batch[rand_idx].cpu())
            Y_tgt_sub_list.append(Y_tgt_batch[rand_idx].cpu())
            
        h1.remove()
        h2.remove()
        
        # 数据移至目标层所在显卡
        H_sub = torch.cat(H_sub_list, dim=0).to(torch.float32).to(layer_device)
        Y_act_sub = torch.cat(Y_act_sub_list, dim=0).to(torch.float32).to(layer_device)
        Y_tgt_sub = torch.cat(Y_tgt_sub_list, dim=0).to(torch.float32).to(layer_device)
        
        W_orig = layer_to_fix.mlp.down_proj.weight.data.to(torch.float32)
        
        Delta_T = Y_tgt_sub - Y_act_sub 
        
        D = H_sub.shape[1]
        HTH = torch.matmul(H_sub.T, H_sub)
        HTT = torch.matmul(H_sub.T, Delta_T)
        
        # 阻尼因子控制正则化
        damp = 0.05 
        damp_val = damp * torch.trace(HTH) / D
        reg_matrix = torch.eye(D, device=layer_device, dtype=torch.float32) * damp_val
        
        A = HTH + reg_matrix
        
        # 求解并叠加增量
        dW_T = torch.linalg.solve(A, HTT)
        layer_to_fix.mlp.down_proj.weight.data.add_(dW_T.T.to(layer_to_fix.mlp.down_proj.weight.dtype))
        
        logging.info(f"✅ 第 {iteration} 轮迭代完成，补偿成功！")
        
        # 清理内存，准备下一轮
        del activations
        del Y_tgt_full
        torch.cuda.empty_cache()

    # ==========================
    # 保存与评测
    # ==========================
    logging.info("\n🎉 所有的迭代剪枝与补偿全部完成！开始最终评测。")
    if not args.continue_saving:
        save_path = os.path.join(args.save_path, args.save_name + f"_i_tgrc_{modelhander.model.config.num_hidden_layers}")
        modelhander.save(path=save_path)
        for da in args.ppl_data:
            ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
            logging.info(f"{da} PPL: {ppl}")
            