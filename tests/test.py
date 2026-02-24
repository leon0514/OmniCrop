import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import omnicrop  # 确保与编译后的模块名一致
import os

# 确保输出目录存在
os.makedirs("test_results", exist_ok=True)

def visualize_case(name, img_w, img_h, boxes, crops, cfg):
    """可视化单个测试用例并保存"""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_facecolor('#f0f0f0') # 灰色背景方便看边界

    # 画布边界
    ax.add_patch(patches.Rectangle((0,0), img_w, img_h, fill=False, edgecolor='black'))

    # 1. 画原始目标框 (红色虚线)
    for b in boxes:
        rect = patches.Rectangle((b.x1, b.y1), b.width, b.height, 
                                 fill=False, edgecolor='red', ls='--', lw=1.5, label='Person')
        ax.add_patch(rect)

    # 2. 画聚类裁剪框 (使用半透明填充)
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f', '#e67e22', '#1abc9c']
    for i, c in enumerate(crops):
        color = colors[i % len(colors)]
        # 填充背景
        rect = patches.Rectangle((c.x1, c.y1), c.width, c.height, 
                                 fill=True, color=color, alpha=0.15)
        # 添加实线边框
        border = patches.Rectangle((c.x1, c.y1), c.width, c.height, 
                                   fill=False, edgecolor=color, lw=2.5, label=f'Crop {i}')
        ax.add_patch(rect)
        ax.add_patch(border)
        
        # 标注Crop ID和尺寸
        ax.text(c.x1 + 5, c.y1 + 25, f"Crop {i}\n{int(c.width)}x{int(c.height)}", 
                color='white', fontsize=8, fontweight='bold', 
                bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

    # 图例去重
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    simple_legend = {k: v for k, v in by_label.items() if k == 'Person' or k == 'Crop 0'}
    ax.legend(simple_legend.values(), simple_legend.keys(), loc='upper right')

    # 显示最新的核心评价参数：crop_count_penalty
    plt.title(f"Test: {name} | Persons: {len(boxes)} | Crops: {len(crops)}\n"
              f"Iterative Best: w_diou={cfg.w_diou}, penalty={cfg.crop_count_penalty}", fontsize=10)
    
    filename = f"test_results/{name}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {filename} - Generated {len(crops)} crops.")

def get_test_cases(img_w, img_h):
    """生成测试数据"""
    cases = []

    # Case 1: 远近场景 (前景大，背景小，测试物理距离合并)
    boxes_1 = [
        omnicrop.BBox(100, 200, 400, 900),   
        omnicrop.BBox(200, 250, 450, 950),   
        omnicrop.BBox(1400, 400, 1450, 520), 
        omnicrop.BBox(1460, 410, 1510, 530), 
        omnicrop.BBox(1600, 400, 1640, 500)  
    ]
    cases.append(("01_Perspective_Mix", boxes_1))

    # Case 2: 极度稀疏 (测试是否会因为 penalty 较小而拒绝合并)
    boxes_2 = [
        omnicrop.BBox(50, 50, 150, 250),
        omnicrop.BBox(img_w-150, 50, img_w-50, 250),
        omnicrop.BBox(50, img_h-300, 150, img_h-100),
        omnicrop.BBox(img_w-150, img_h-300, img_w-50, img_h-100)
    ]
    cases.append(("02_Sparse_Corners", boxes_2))

    # Case 3: 密集人群 (测试重叠合并逻辑)
    boxes_3 = []
    start_x, start_y = 600, 200
    for i in range(5):
        for j in range(4):
            bx = start_x + i * 80 + random.randint(-5, 5)
            by = start_y + j * 160 + random.randint(-5, 5)
            boxes_3.append(omnicrop.BBox(bx, by, bx+60, by+140))
    cases.append(("03_Dense_Crowd", boxes_3))

    # Case 4: 随机杂乱场景 (鲁棒性测试)
    boxes_4 = []
    for _ in range(15):
        w = random.randint(60, 180)
        h = int(w * random.uniform(2.0, 3.5))
        x = random.randint(100, img_w - 200)
        y = random.randint(100, img_h - 400)
        boxes_4.append(omnicrop.BBox(x, y, x+w, y+h))
    cases.append(("04_Random_Stress", boxes_4))

    return cases

def run_tests():
    # 1. 引擎初始化
    engine = omnicrop.OmniCropEngine(max_crop_size=300, padding=40)
    
    # 2. 配置优化参数
    cfg = omnicrop.Config()
    
    # --- 核心改动：不再使用阈值，改用惩罚项 ---
    # crop_count_penalty 是“平衡木”。
    # 调大（如 100.0）：极其讨厌多个框，会尽量把东西全塞进一个框里。
    # 调小（如 10.0）：非常在乎分辨率利用率，会生成很多刚好包裹住目标的小框。
    cfg.crop_count_penalty = 200.0  
    
    # 距离权重和面积扩张权重（用于确定合并的优先级顺序）
    cfg.w_diou = 10.0
    cfg.w_expansion = 5.0
    
    # NMS 阈值依然保留，用于最后物理重叠框的融合
    cfg.nms_threshold = 0.3
    
    # 是否强制比例
    cfg.enable_aspect_ratio_fix = False
    cfg.target_aspect_ratio = 1.0 # 1:1

    img_w, img_h = 1920, 1080

    # 3. 获取测试数据并执行
    test_cases = get_test_cases(img_w, img_h)

    print(f"=== OmniCrop 迭代最优引擎测试 ===")
    print(f"配置: Penalty={cfg.crop_count_penalty}, DiouW={cfg.w_diou}, ExpandW={cfg.w_expansion}")

    for name, boxes in test_cases:
        try:
            # 调用 C++ 接口，它会自动遍历合并路径并挑选得分最高的状态
            crops = engine.cluster_and_crop(boxes, img_w, img_h, cfg)
            
            visualize_case(name, img_w, img_h, boxes, crops, cfg)
        except Exception as e:
            print(f"[ERROR] Case {name} 发生故障: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[FINISH] 所有测试完成，请查看 'test_results/' 目录。")

if __name__ == "__main__":
    run_tests()