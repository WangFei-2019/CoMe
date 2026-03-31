import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import pandas as pd
from matplotlib.colors import LogNorm

# 设置高品质学术画图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze(pt_file, tokenizer_path):
    print(f"Loading data from {pt_file}...")
    data = torch.load(pt_file)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    tokens = data["tokens"].numpy()
    ce_gold = data["ce_gold"].float().numpy()
    ce_uncomp = data["ce_uncomp"].float().numpy()
    ce_comp = data["ce_comp"].float().numpy()
    
    diff_uncomp = ce_uncomp - ce_gold
    diff_comp = ce_comp - ce_gold
    
    records = []
    for i in range(len(tokens)):
        token_str = tokenizer.convert_ids_to_tokens(int(tokens[i]))
        position = "Word Start" if token_str.startswith(' ') else "Subword / Punct"
        clean_str = token_str.replace(' ', '')
            
        records.append({
            "Token": clean_str,
            "Position": position,
            "Uncomp_Diff": diff_uncomp[i],
            "Comp_Diff": diff_comp[i],
        })
    return pd.DataFrame(records)

def visualize_academic_triptych(df, save_path="academic_degradation_analysis.png"):
    """绘制 1x3 的学术级三联图"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # ---------------------------------------------------------
    # Panel 1: SymLog Hexbin Density Plot (Concern 1-4 Fixed)
    # ---------------------------------------------------------
    ax1 = axes[0]
    
    x_data = df['Uncomp_Diff'].astype(float).to_numpy()
    y_data = df['Comp_Diff'].astype(float).to_numpy()
    
    # Calculate counts Above vs Below y=x (FDD-Ridge makes it worse vs better)
    diff_data = y_data - x_data
    num_worse = len(diff_data[diff_data > 0])
    num_better = len(diff_data[diff_data <= 0])
    total = len(df)

    # 1. 🌟 [Fix Concerned Limits Logic]: Clamp view based on data density, not abs max.
    # We cap max view based on 99.9th percentile + buffer, and use actual min.
    capped_max = max(np.percentile(x_data, 99.9), np.percentile(y_data, 99.9))
    actual_min = min(x_data.min(), y_data.min())
    
    # Smart range: Ensure we see at least up to 1.0, and actual min minus buffer.
    view_max = max(capped_max * 1.1, 1.0) 
    view_min = min(actual_min - 0.1, -0.2) 

    # 2. 🌟 [Fix Overplotting Concentration]: Switch A to SymLog Scale.
    # This expands center center while handling sparse outliers.
    ax1.set_xscale('symlog', linthresh=0.1)
    ax1.set_yscale('symlog', linthresh=0.1)

    # Use smaller gridsize with symlog to better discern center多寡
    hb = ax1.hexbin(x_data, y_data, gridsize=30, cmap='Blues', bins='log', mincnt=1)
    cb = fig.colorbar(hb, ax=ax1, orientation='vertical', pad=0.02)
    cb.set_label('Log(Token Count)', rotation=270, labelpad=15)
    
    # Baselines (Respect actual data boundaries for line drawing)
    data_max_true = max(x_data.max(), y_data.max())
    data_min_true = min(x_data.min(), y_data.min())
    ax1.plot([data_min_true - 1, data_max_true + 1], [data_min_true - 1, data_max_true + 1], 'r--', lw=2, label='y=x (No Improvement)')
    ax1.axhline(0, color='green', linestyle='--', lw=2, label='y=0 (Perfect Repair)')
    
    # Apply Capped View Limits ( Concern 3 & 4 fixed)
    ax1.set_xlim(view_min, view_max)
    ax1.set_ylim(view_min, view_max)

    ax1.set_title("A. Density of Degradation (SymLog Grid)", fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel("Δ CE Loss (Direct Pruning)", fontsize=12)
    ax1.set_ylabel("Δ CE Loss (FDD-Ridge Compensation)", fontsize=12)
    
    # 3. 🌟 [Address Concern: Above vs Below]: Explicitly annotate counts in academic style.
    annotation_text = f"Pruning error increased: {num_worse/total:.1%}\nPruning error decreased: {num_better/total:.1%}"
    # Standard transAxes transform places text in top-left relative to axis box
    ax1.text(0.02, 0.98, annotation_text, 
             transform=ax1.transAxes, va='top', ha='left', fontsize=11, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    ax1.legend(loc='lower right')

    # ---------------------------------------------------------
    # Panel 2: Violin Plot Distribution
    # ---------------------------------------------------------
    ax2 = axes[1]
    df_melt = pd.melt(df, value_vars=['Uncomp_Diff', 'Comp_Diff'], 
                      var_name='Method', value_name='CE_Loss_Diff')
    df_melt['Method'] = df_melt['Method'].replace({'Uncomp_Diff': 'Direct Pruning', 'Comp_Diff': 'FDD-Ridge'})
    
    # 🌟 修复 FutureWarning: 添加 hue='Method', legend=False
    sns.violinplot(data=df_melt, x='Method', y='CE_Loss_Diff', ax=ax2, hue='Method', palette="muted", inner="quartile", cut=0, legend=False)
    
    ax2.set_title("B. Error Distribution Shift", fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel("", fontsize=12)
    ax2.set_ylabel("Δ CE Loss (Error Magnitude)", fontsize=12)
    
    ax2.annotate('Catastrophic\nTail Erased', xy=(0.0, df['Uncomp_Diff'].max() * 0.8), 
                 xytext=(0.5, df['Uncomp_Diff'].max() * 0.9),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=11, ha='center', color='darkred')

    # ---------------------------------------------------------
    # Panel 3: Rescue Trajectory Slope Graph
    # ---------------------------------------------------------
    ax3 = axes[2]
    top_worst = df.nlargest(12, 'Uncomp_Diff').copy()
    
    tokens = [str(t).replace(' ', '').replace('_', '').replace('\u2581', '') for t in top_worst['Token'].values]
    y1_vals = top_worst['Uncomp_Diff'].values
    y2_vals = top_worst['Comp_Diff'].values
    
    text_y_coords = np.copy(y1_vals)
    min_spacing = 0.15 
    
    sort_idx = np.argsort(text_y_coords)
    for i in range(1, len(sort_idx)):
        curr_idx = sort_idx[i]
        prev_idx = sort_idx[i-1]
        if text_y_coords[curr_idx] - text_y_coords[prev_idx] < min_spacing:
            text_y_coords[curr_idx] = text_y_coords[prev_idx] + min_spacing

    for i in range(len(y1_vals)):
        y1, y2, txt, ty = y1_vals[i], y2_vals[i], tokens[i], text_y_coords[i]
        color = 'forestgreen' if y2 < y1 else 'crimson'
        
        ax3.plot([1, 2], [y1, y2], marker='o', color=color, markersize=6, lw=2, alpha=0.7)
        ax3.text(0.9, ty, txt, ha='right', va='center', fontsize=11, fontweight='bold')
        
        if abs(ty - y1) > 0.01:
            ax3.plot([0.92, 1.0], [ty, y1], color='gray', linestyle=':', lw=1, alpha=0.6)
            
    ax3.set_xlim(0.3, 2.5) 
    
    max_y_plot = max(text_y_coords.max(), y2_vals.max()) + 0.2
    min_y_plot = min(y1_vals.min(), y2_vals.min()) - 0.2
    ax3.set_ylim(min_y_plot, max_y_plot)
    
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['Direct Pruning\n(Before)', 'FDD-Ridge\n(After)'], fontsize=12)
    ax3.set_ylabel("Δ CE Loss", fontsize=12)
    ax3.set_title("C. Rescue Trajectory of Top-12 Worst Tokens", fontsize=14, fontweight='bold', pad=10)
    
    sns.despine(ax=ax3, left=True)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    # ---------------------------------------------------------
    # 整体布局调整与保存
    # ---------------------------------------------------------
    plt.suptitle("Microscopic Interpretability: How FDD-Ridge Fixes Iterative Pruning", fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Academic Triptych Visualization saved to: {save_path}")

if __name__ == "__main__":
    pt_file_path = "fdd_ridge_llama_22L/token_diag_Layer17.pt" 
    model_path = "/workspace/wangfei154/models/meta-llama/llama-2-7b-hf"
    
    if os.path.exists(pt_file_path):
        df_results = load_and_analyze(pt_file_path, model_path)
        visualize_academic_triptych(df_results, save_path="academic_triptych_analysis.png")
    else:
        print(f"Error: File {pt_file_path} not found.")