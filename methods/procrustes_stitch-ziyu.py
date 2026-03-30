import os
import json
import torch
import numpy as np
from tqdm import tqdm
import logging

from utils.data_utils import get_trainloaders

@torch.inference_mode()
def main_func(args, modelhander):
    logging.info("🚀 启动【误差传播轨迹探针 2.0】(数据截断修复版)...")
    
    # 0. 健康检查：确保权重没有被污染
    for name, param in modelhander.model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            logging.error(f"❌ 警告：检测到模型权重 {name} 中存在 nan 或 inf！请重启环境并重新加载模型。")
            return

    # 1. 获取校准数据 (使用原始安全参数保证数据完整生成)
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
    
    # 算出现有的真实样本总数，我们只取前 8 个作为探针
    total_nsamples = testenc.numel() // seqlen
    probe_nsamples = min(8, total_nsamples)
    logging.info(f"✅ 数据加载成功，总样本 {total_nsamples} 个，探针将抽取前 {probe_nsamples} 个样本进行分析。")

    # 2. 记录 Teacher 轨迹
    logging.info("1/3 正在记录原模型 (32层) 的特征轨迹...")
    teacher_hidden_states = []
    for i in tqdm(range(probe_nsamples), desc="Teacher Forward"):
        inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(device)
        # 强制使用 float32 提取特征以防溢出
        outputs = modelhander.model(inputs, output_hidden_states=True)
        teacher_hidden_states.append([h.detach().cpu().float() for h in outputs.hidden_states])

    # 3. 执行剪枝决策 (这里沿用你之前的 10 层索引)
    prune_indices = [29, 28, 27, 26, 25, 24, 23, 21, 12, 11]
    prune_indices.sort(reverse=True)
    
    surviving_original_indices = [i for i in range(32) if i not in prune_indices]
    
    logging.info(f"2/3 正在物理移除层: {prune_indices}")
    for prune_idx in prune_indices:
        modelhander.remove_layers([prune_idx])
    
    # 4. 记录 Student 轨迹并对比
    logging.info("3/3 正在对比剪枝前后的特征点分布...")
    l2_errors = {j: [] for j in range(len(surviving_original_indices) + 1)}
    cos_sims = {j: [] for j in range(len(surviving_original_indices) + 1)}

    for i in tqdm(range(probe_nsamples), desc="Student Forward & Compare"):
        inputs = testenc[:, (i * seqlen):((i+1) * seqlen)].to(device)
        outputs = modelhander.model(inputs, output_hidden_states=True)
        student_h_states = [h.detach().cpu().float() for h in outputs.hidden_states]
        
        for j, h_student in enumerate(student_h_states):
            # 找到在 Teacher 中对应的原始层索引
            orig_k = 0 if j == 0 else surviving_original_indices[j - 1] + 1
            h_teacher = teacher_hidden_states[i][orig_k]
            
            # 计算 L2 相对误差 (归一化防止尺度干扰)
            diff_norm = torch.norm(h_student - h_teacher, p=2, dim=-1)
            teacher_norm = torch.norm(h_teacher, p=2, dim=-1) + 1e-9
            rel_l2 = (diff_norm / teacher_norm).mean().item()
            
            # 计算 Cosine Similarity
            cos_sim = torch.nn.functional.cosine_similarity(h_student, h_teacher, dim=-1).mean().item()
            
            l2_errors[j].append(rel_l2)
            cos_sims[j].append(cos_sim)

    # 5. 输出报告
    logging.info("\n" + "="*70)
    logging.info(f"{'当前层':<8} | {'原对应层':<10} | {'相对 L2 误差':<15} | {'CosSim 方向保留度'}")
    logging.info("-" * 70)
    
    report_data = {}
    for j in range(len(student_h_states)):
        orig_k = 0 if j == 0 else surviving_original_indices[j - 1] + 1
        avg_l2 = np.mean(l2_errors[j])
        avg_cos = np.mean(cos_sims[j])
        
        # 标记断点
        is_break = ""
        if j > 0 and j < len(student_h_states)-1:
            if surviving_original_indices[j-1] != (surviving_original_indices[j-2]+1 if j>1 else 0):
                is_break = " ⚠️断点"
        if j == 12: is_break = " ⚠️断点" # 根据剪枝列表硬编码标记第一个大断点

        logging.info(f"L{j:<2} | Orig {orig_k:<4} {is_break:<6} | {avg_l2:<14.6f} | {avg_cos:.6f}")
        report_data[f"L{j}_Orig{orig_k}"] = {"Rel_L2": float(avg_l2), "CosSim": float(avg_cos)}
        
    logging.info("="*70)
    
    with open("error_propagation_report.json", "w") as f:
        json.dump(report_data, f, indent=4)
    logging.info("✅ 探测数据已保存至 error_propagation_report.json")