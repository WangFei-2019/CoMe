import os
import json
import torch
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict

from utils.data_utils import get_trainloaders

@torch.inference_mode()
def main_func(args, modelhander):
    logging.info("🚀 启动【全息误差演化热力图探针 (Holographic Evolution Heatmap)】...")
    
    # 1. 准备校准数据
    dataloader = get_trainloaders(
        args.calibration_dataset,
        tokenizer=modelhander.tokenizer,
        nsamples=128, 
        seed=args.seed,
        seqlen=modelhander.model.seqlen
    )
    device = modelhander.model.device
    testenc = dataloader.input_ids
    seqlen = modelhander.model.seqlen
    probe_nsamples = 4

    # 2. 获取 Teacher 完整特征基准
    logging.info("1/2 正在记录原模型 (Teacher) 全局参考特征...")
    teacher_hiddens = []
    for i in tqdm(range(probe_nsamples), desc="Teacher Forward"):
        inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(device)
        outputs = modelhander.model(inputs, output_hidden_states=True, use_cache=False)
        teacher_hiddens.append([h.detach().cpu().float() for h in outputs.hidden_states])

    layer_num = modelhander.model.config.num_hidden_layers
    # 核心修复1：hidden_states 长度是 33 (包含 Embedding 后的初态)
    total_states = layer_num + 1 
    
    # 初始化三个 2D 热力图矩阵，尺寸改为 33x33
    heatmap_l2 = np.full((total_states, total_states), np.nan)
    heatmap_cos = np.full((total_states, total_states), np.nan)
    heatmap_norm_ratio = np.full((total_states, total_states), np.nan)

    logging.info("2/2 正在构建轨迹热力图矩阵 (逐层切除与全息追踪)...")
    
    for target_l in tqdm(range(1, layer_num - 1), desc="Pruning & Tracing"):
        layer_backup = modelhander.model.model.layers[target_l]
        modelhander.model.model.layers.pop(target_l)
        
        try:
            # 核心修复2：使用 defaultdict 自动处理动态生成的层索引
            batch_l2 = defaultdict(list)
            batch_cos = defaultdict(list)
            batch_norm = defaultdict(list)
            
            for i in range(probe_nsamples):
                inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(device)
                outputs = modelhander.model(inputs, output_hidden_states=True, use_cache=False)
                student_hiddens = [h.detach().cpu().float() for h in outputs.hidden_states]
                
                for stu_idx in range(target_l, len(student_hiddens)):
                    tea_idx = stu_idx + 1 # 物理对齐
                    if tea_idx >= total_states:
                        continue # 安全保护
                        
                    h_stu = student_hiddens[stu_idx]
                    h_tea = teacher_hiddens[i][tea_idx]
                    
                    diff_norm = torch.norm(h_stu - h_tea, p=2, dim=-1)
                    tea_norm = torch.norm(h_tea, p=2, dim=-1) + 1e-9
                    rel_l2 = (diff_norm / tea_norm).mean().item()
                    
                    cos_sim = torch.nn.functional.cosine_similarity(h_stu, h_tea, dim=-1).mean().item()
                    
                    stu_norm = torch.norm(h_stu, p=2, dim=-1).mean().item()
                    tea_norm_mean = tea_norm.mean().item()
                    norm_ratio = stu_norm / tea_norm_mean
                    
                    obs_l = tea_idx
                    batch_l2[obs_l].append(rel_l2)
                    batch_cos[obs_l].append(cos_sim)
                    batch_norm[obs_l].append(norm_ratio)
                    
            # 写入热力图矩阵
            for obs_l, l2_vals in batch_l2.items():
                heatmap_l2[target_l, obs_l] = np.mean(l2_vals)
                heatmap_cos[target_l, obs_l] = np.mean(batch_cos[obs_l])
                heatmap_norm_ratio[target_l, obs_l] = np.mean(batch_norm[obs_l])
                
        finally:
            # 无论发生什么，都要把层装回去，保证物理环境干净
            modelhander.model.model.layers.insert(target_l, layer_backup)

    # 3. 保存数据
    res = {
        "heatmap_l2": heatmap_l2.tolist(),
        "heatmap_cos": heatmap_cos.tolist(),
        "heatmap_norm_ratio": heatmap_norm_ratio.tolist()
    }
    with open("heatmap_data_matrices.json", "w") as f:
        json.dump(res, f)
    
    logging.info("✅ 矩阵探测完毕！已保存至 heatmap_data_matrices.json")