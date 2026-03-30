import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_holographic_heatmaps(json_path="heatmap_data_matrices.json", save_path="pruning_heatmaps.pdf"):
    print("📊 正在加载热力图数据...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 转换为 NumPy 矩阵
    mat_l2 = np.array(data["heatmap_l2"])
    mat_cos = np.array(data["heatmap_cos"])
    mat_norm = np.array(data["heatmap_norm_ratio"])

    # 设置全局绘图风格 (学术风)
    sns.set_theme(style="white")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # 创建 1行3列 的大图板
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), gridspec_kw={'wspace': 0.15})
    
    # 共同的坐标轴设置参数
    # 因为我们测了 Layer 1 到 Layer 30，为了美观可以裁剪掉全空的行列
    # 真实数据在 Y:[1~30], X:[2~32]
    # 我们直接画完整的 33x33，NaN 会自动显示为空白背景
    ticks = np.arange(0, 33, 4)  # 每隔 4 层标一个刻度，避免太挤
    
    # ---------------------------------------------------------
    # 图 1: L2 Relative Error (绝对误差)
    # ---------------------------------------------------------
    # 误差越小越好，使用红黄白配色 (越红误差越大)
    sns.heatmap(mat_l2, ax=axes[0], cmap="YlOrRd", 
                cbar_kws={'label': 'Relative L2 Error'},
                vmin=0.0, vmax=1.0, # 限制最大值以防个别异常点把颜色带偏
                mask=np.isnan(mat_l2)) # 屏蔽 NaN 值
    axes[0].set_title("(a) L2 Error Evolution\n(Redder = Higher Error)", fontsize=16, fontweight='bold')
    
    # ---------------------------------------------------------
    # 图 2: Cosine Similarity (语义方向)
    # ---------------------------------------------------------
    # 相似度越接近 1 越好，使用冷色调 (越深蓝相似度越高)
    sns.heatmap(mat_cos, ax=axes[1], cmap="mako_r", 
                cbar_kws={'label': 'Cosine Similarity'},
                vmin=0.5, vmax=1.0, 
                mask=np.isnan(mat_cos))
    axes[1].set_title("(b) Semantic Direction (CosSim)\n(Darker = Better Alignment)", fontsize=16, fontweight='bold')
    
    # ---------------------------------------------------------
    # 图 3: Norm Ratio (幅度能量比例)
    # ---------------------------------------------------------
    # 理想值是 1.0。使用发散型色谱 (蓝-白-红)，白色代表 1.0
    # <1 表示能量萎缩(偏蓝)，>1 表示能量发散(偏红)
    sns.heatmap(mat_norm, ax=axes[2], cmap="coolwarm", 
                cbar_kws={'label': 'Norm Ratio (Student / Teacher)'},
                vmin=0.5, vmax=1.5, center=1.0,
                mask=np.isnan(mat_norm))
    axes[2].set_title("(c) Energy Magnitude Ratio\n(White = 1.0, Stable Energy)", fontsize=16, fontweight='bold')

    # ---------------------------------------------------------
    # 统一设置坐标轴标签
    # ---------------------------------------------------------
    for ax in axes:
        ax.set_ylabel("Pruned Layer Index (Where trauma happens)", fontsize=14)
        ax.set_xlabel("Observation Layer Index (Where signal arrives)", fontsize=14)
        ax.set_xticks(ticks + 0.5)
        ax.set_xticklabels(ticks, rotation=0)
        ax.set_yticks(ticks + 0.5)
        ax.set_yticklabels(ticks, rotation=0)
        # 翻转 Y 轴，让 Layer 0 在最上面，符合网络深度直觉
        ax.invert_yaxis()

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 热力图已成功保存为: {save_path}")
    
    # 如果在 Jupyter 中运行，取消下面这行的注释可以直接显示
    # plt.show()

if __name__ == "__main__":
    plot_holographic_heatmaps()