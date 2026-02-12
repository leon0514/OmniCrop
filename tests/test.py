import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import omnicrop

def run_test():
    # 1. 引擎初始化
    engine = omnicrop.OmniCropEngine(max_crop_size=1280, padding=50)
    cfg = omnicrop.Config()
    cfg.enable_aspect_ratio_fix = True
    cfg.target_aspect_ratio = 1.0  # 强制正方形
    cfg.w_scale = 8.0              # 防止远近目标误合并

    # 2. 模拟图像数据
    img_w, img_h = 1920, 1080
    person_boxes = [
        # 群组 A (近处)
        omnicrop.BBox(400, 500, 550, 900),
        omnicrop.BBox(450, 520, 600, 920),
        # 群组 B (远处)
        omnicrop.BBox(1200, 200, 1240, 300),
        omnicrop.BBox(1250, 210, 1290, 310)
    ]

    # 3. 执行聚类与裁剪
    crops = engine.cluster_and_crop(person_boxes, img_w, img_h, cfg)

    # 4. 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0) # y轴反转

    # 画原始框
    for b in person_boxes:
        ax.add_patch(patches.Rectangle((b.x1, b.y1), b.width, b.height, 
                                     fill=False, edgecolor='red', ls='--'))
    
    # 画输出切片
    for i, c in enumerate(crops):
        ax.add_patch(patches.Rectangle((c.x1, c.y1), c.width, c.height, 
                                     fill=True, color='green', alpha=0.2, lw=2))
        print(f"Crop {i}: Center({c.centerX():.1f}, {c.centerY():.1f}) Size({c.width()}x{c.height()})")

    plt.title("OmniCrop Engine v2.0 Clustering Test")
    plt.savefig("test_result.png")
    print("\n[SUCCESS] 测试结果已保存为 'test_result.png'")
    # plt.show() # 如果在服务器上运行，请注释掉此行

if __name__ == "__main__":
    run_test()