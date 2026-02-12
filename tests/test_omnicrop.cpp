#include "OmniCrop.hpp"
#include <cassert>
#include <iostream>
#include <vector>

void test_basic_clustering() {
    OmniCropEngine engine(1280, 50, 5, 3.5f);
    OmniCropEngine::Config cfg;
    cfg.enable_aspect_ratio_fix = false;

    // 模拟两个非常接近且尺度一致的框
    std::vector<BBox> boxes = {
        {100, 100, 200, 200},
        {110, 110, 210, 210}
    };

    auto results = engine.cluster_and_crop(boxes, 2000, 2000, cfg);
    
    // 预期：这两个框应该合并成一个
    assert(results.size() == 1);
    std::cout << "[Test Pass] Basic clustering works." << std::endl;
}

void test_scale_consistency() {
    OmniCropEngine engine(1280, 50, 5, 3.5f);
    OmniCropEngine::Config cfg;
    cfg.w_scale = 10.0f; // 强化尺度一致性

    // 一个大目标，一个小目标，位置接近
    std::vector<BBox> boxes = {
        {100, 100, 500, 900}, // 大
        {450, 450, 480, 510}  // 小
    };

    auto results = engine.cluster_and_crop(boxes, 2000, 2000, cfg);
    
    // 预期：由于高度差巨大，Scale Loss 很高，不应合并
    assert(results.size() == 2);
    std::cout << "[Test Pass] Scale consistency prevented bad merge." << std::endl;
}

void test_aspect_ratio_fix() {
    OmniCropEngine engine(1280, 50, 5, 3.5f);
    OmniCropEngine::Config cfg;
    cfg.enable_aspect_ratio_fix = true;
    cfg.target_aspect_ratio = 1.0f; // 强制正方形

    std::vector<BBox> boxes = {
        {100, 100, 300, 200} // 宽 200, 高 100
    };

    auto results = engine.cluster_and_crop(boxes, 2000, 2000, cfg);
    
    // 预期：输出应该是正方形
    float w = results[0].width();
    float h = results[0].height();
    assert(std::abs(w - h) < 1e-3);
    std::cout << "[Test Pass] Aspect ratio correction works." << std::endl;
}

int main() {
    try {
        test_basic_clustering();
        test_scale_consistency();
        test_aspect_ratio_fix();
        std::cout << "\nAll C++ Tests Passed Successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}