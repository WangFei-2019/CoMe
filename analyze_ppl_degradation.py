import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import pandas as pd

# 设置画图风格
sns.set_theme(style="whitegrid")
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 支持中文
# plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze(pt_file, tokenizer_path):
    print(f"Loading data from {pt_file}...")
    data = torch.load(pt_file)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    tokens = data["tokens"].numpy()
    ce_gold = data["ce_gold"].float().numpy()
    ce_uncomp = data["ce_uncomp"].float().numpy()
    ce_comp = data["ce_comp"].float().numpy()

    if "prob_gold" in data:
        prob_gold = data["prob_gold"].float().numpy()
        prob_uncomp = data["prob_uncomp"].float().numpy()
        prob_comp = data["prob_comp"].float().numpy()
    
    # 差值计算：正数代表退化（Loss变大）
    diff_uncomp = ce_uncomp - ce_gold
    diff_comp = ce_comp - ce_gold
    
    records = []
    for i in range(len(tokens)):
        token_id = tokens[i]
        token_str = tokenizer.convert_ids_to_tokens(int(token_id))
        
        # 1. 判定 Token 在单词中的位置
        # Llama tokenizer 使用 ' ' (U+2581) 表示一个新单词的开始
        if token_str.startswith(' '):
            position = "Word Start" # (单词起始)
            clean_str = token_str.replace(' ', '')
        elif token_str in [',', '.', '?', '!', ':', ';', '\n']:
            position = "Punctuation" # (标点符号)
            clean_str = token_str
        else:
            position = "Subword" # (词中/词尾)
            clean_str = token_str
            
        # 您还可以引入 NLTK 在这里对 clean_str 进行词性标注 (POS Tagging)
        
        records.append({
            "Token": clean_str,
            "Position": position,
            "Gold_CE": ce_gold[i],
            "Uncomp_Diff": diff_uncomp[i],
            "Comp_Diff": diff_comp[i],
        })
        
    df = pd.DataFrame(records)
    return df

def cross_comparison_analysis(df, top_n=20):
    print(f"\n{'='*80}")
    print(f"⚔️ 交叉对比分析 (Cross-Comparison of Degradation)")
    print(f"{'='*80}")
    
    # 1. 直接剪枝中最差的 Top N，在我们的补偿中怎么样？
    worst_uncomp = df.nlargest(top_n, 'Uncomp_Diff')
    print(f"\n🚨 [直接剪枝] 最致命的 {top_n} 个 Token，在 [FDD-Ridge 补偿] 下的表现：")
    print(f"{'Token':<15} | {'Type':<22} | {'Δ 直接剪枝 (坏)':<15} | {'Δ FDD补偿 (好/坏)':<15}")
    print("-" * 75)
    for _, row in worst_uncomp.iterrows():
        status = "✅ 完美修复" if row['Comp_Diff'] < 0.5 else "⚠️ 部分修复"
        print(f"{row['Token']:<15} | {row['Position']:<22} | +{row['Uncomp_Diff']:<14.3f} | +{row['Comp_Diff']:<14.3f} ({status})")

    # 2. 我们补偿后最差的 Top N，在直接剪枝中怎么样？
    worst_comp = df.nlargest(top_n, 'Comp_Diff')
    print(f"\n🚨 [FDD-Ridge 补偿] 最致命的 {top_n} 个 Token，在 [直接剪枝] 下的表现：")
    print(f"{'Token':<15} | {'Type':<22} | {'Δ FDD补偿 (坏)':<15} | {'Δ 直接剪枝 (好/坏)':<15}")
    print("-" * 75)
    for _, row in worst_comp.iterrows():
        reason = "🎯 自信税/平滑诅咒" if row['Uncomp_Diff'] < row['Comp_Diff'] else "💥 绝症词(原版也错)"
        print(f"{row['Token']:<15} | {row['Position']:<22} | +{row['Comp_Diff']:<14.3f} | +{row['Uncomp_Diff']:<14.3f} ({reason})")

def visualize_degradation(df, save_path="degradation_plot.png"):
    # Create a 1x2 subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # We add a slight jitter or just rely on alpha to see density
    max_x = df['Uncomp_Diff'].max()
    max_y = df['Comp_Diff'].max()
    global_max = max(max_x, max_y) * 1.05
    min_val = min(df['Uncomp_Diff'].min(), df['Comp_Diff'].min(), -0.1)

    # ==========================================
    # Subplot 1: Global View with SymLog Scale
    # ==========================================
    ax1 = axes[0]
    sns.scatterplot(
        data=df, x='Uncomp_Diff', y='Comp_Diff', hue='Position', 
        alpha=0.6, edgecolor=None, s=40, ax=ax1
    )
    
    # Baselines
    ax1.plot([min_val, global_max], [min_val, global_max], 'r--', lw=2, label='y=x (Equally Degraded)')
    ax1.axhline(0, color='green', linestyle='--', lw=2, label='y=0 (Perfectly Repaired)')
    
    # Use Symmetrical Log Scale to handle the extreme outliers without squashing the center
    ax1.set_xscale('symlog', linthresh=0.5)
    ax1.set_yscale('symlog', linthresh=0.5)
    
    ax1.set_title("Global View (SymLog Scale)", fontsize=15, pad=15)
    ax1.set_xlabel("Δ CE Loss (Uncompensated Pruning)", fontsize=13)
    ax1.set_ylabel("Δ CE Loss (FDD-Ridge Compensation)", fontsize=13)
    ax1.legend(loc='upper left')

    # Annotate Top 5 outliers in Global View
    top_outliers = df.nlargest(5, 'Uncomp_Diff')
    for _, row in top_outliers.iterrows():
        ax1.annotate(
            row['Token'], 
            (row['Uncomp_Diff'], row['Comp_Diff']),
            xytext=(5, 5), textcoords='offset points', 
            fontsize=10, color='darkred', fontweight='bold'
        )

    # ==========================================
    # Subplot 2: Zoomed-in View (Linear Scale)
    # ==========================================
    ax2 = axes[1]
    sns.scatterplot(
        data=df, x='Uncomp_Diff', y='Comp_Diff', hue='Position', 
        alpha=0.5, edgecolor=None, s=40, ax=ax2
    )
    
    # Zoom range: focus on the dense cluster (e.g., -0.2 to 2.0)
    zoom_limit = min(2.0, global_max) 
    
    ax2.plot([min_val, zoom_limit], [min_val, zoom_limit], 'r--', lw=2, label='y=x')
    ax2.axhline(0, color='green', linestyle='--', lw=2, label='y=0')
    
    ax2.set_xlim(-0.2, zoom_limit)
    ax2.set_ylim(-0.2, zoom_limit)
    
    ax2.set_title(f"Zoomed-in View (Dense Region: < {zoom_limit})", fontsize=15, pad=15)
    ax2.set_xlabel("Δ CE Loss (Uncompensated Pruning)", fontsize=13)
    ax2.set_ylabel("Δ CE Loss (FDD-Ridge Compensation)", fontsize=13)
    ax2.legend(loc='upper left')

    # Main Title
    plt.suptitle("Token-level CE Loss Degradation: Uncompensated vs. FDD-Ridge", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Visualization saved to: {save_path}")

if __name__ == "__main__":
    # 替换为您保存的 pt 文件路径和模型路径
    pt_file_path = "./fdd_ridge_llama_22L/token_diag_Layer27.pt" 
    model_path = "/workspace/wangfei154/models/meta-llama/llama-2-7b-hf"
    
    df_results = load_and_analyze(pt_file_path, model_path)
    
    # 交叉分析打印
    cross_comparison_analysis(df_results, top_n=15)
    
    # 画图并保存
    visualize_degradation(df_results, save_path="token_degradation_scatter.png")