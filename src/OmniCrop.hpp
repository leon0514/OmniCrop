#ifndef OMNICROP_HPP
#define OMNICROP_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <queue>

/**
 * @brief 基础边界框结构体
 */
struct BBox {
    float x1, y1, x2, y2;
    BBox() : x1(0), y1(0), x2(0), y2(0) {}
    BBox(float _x1, float _y1, float _x2, float _y2) : x1(_x1), y1(_y1), x2(_x2), y2(_y2) {}
    
    inline float width() const { return x2 - x1; }
    inline float height() const { return y2 - y1; }
    inline float area() const { return (x2 - x1) * (y2 - y1); }
    inline float centerX() const { return (x1 + x2) * 0.5f; }
    inline float centerY() const { return (y1 + y2) * 0.5f; }
};

/**
 * @brief 聚类配置项 (移出类外以修复编译错误)
 */
struct OmniCropConfig {
    float w_size = 2.0f;
    float w_density = 3.0f;
    float w_scale = 4.0f;
    float w_square = 1.5f;
    float w_alignment = 0.5f;

    float min_obj_size = 20.0f;
    float max_obj_size = 1000.0f;

    bool enable_aspect_ratio_fix = true;
    float target_aspect_ratio = 1.0f;
};

/**
 * @brief 聚类簇结构体 (添加显式构造函数)
 */
struct Cluster {
    int id;
    BBox bbox;
    float area_sum;
    bool active;

    Cluster(int _id, BBox _bbox, float _area_sum) 
        : id(_id), bbox(_bbox), area_sum(_area_sum), active(true) {}
};

struct MergeCandidate {
    float loss;
    int u, v;
    bool operator>(const MergeCandidate& other) const { return loss > other.loss; }
};

/**
 * @brief OmniCropEngine: 修复了默认初始化器编译问题的版本
 */
class OmniCropEngine {
public:
    // 为了保持接口兼容，定义别名
    using Config = OmniCropConfig;

    OmniCropEngine(int max_crop_size = 1280, int padding = 50, int max_outputs = 5, float stop_threshold = 3.5f)
        : max_crop_size_(max_crop_size), padding_(padding), max_outputs_(max_outputs), stop_threshold_(stop_threshold) {}

    std::vector<BBox> cluster_and_crop(const std::vector<BBox>& person_boxes, int img_w, int img_h, Config cfg = Config()) {
        if (person_boxes.empty()) return {};

        std::vector<Cluster> clusters;
        std::vector<BBox> direct_outputs;

        // 1. 预处理
        for (const auto& box : person_boxes) {
            float w = box.width(), h = box.height();
            if (w < cfg.min_obj_size || h < cfg.min_obj_size) continue;
            if (w > cfg.max_obj_size || h > cfg.max_obj_size) {
                direct_outputs.push_back(finalize_box(box, img_w, img_h, cfg));
                continue;
            }
            // 使用构造函数而非大括号聚合初始化
            clusters.push_back(Cluster((int)clusters.size(), box, box.area()));
        }

        int cn = (int)clusters.size();
        if (cn == 0) return direct_outputs;

        // 2. 堆初始化
        std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, std::greater<MergeCandidate>> pq;
        for (int i = 0; i < cn; ++i) {
            for (int j = i + 1; j < cn; ++j) {
                float loss = calculate_loss(clusters[i], clusters[j], cfg);
                if (loss < std::numeric_limits<float>::infinity()) pq.push({loss, i, j});
            }
        }

        // 3. 聚类
        int active_count = cn;
        while (active_count > 1 && !pq.empty()) {
            MergeCandidate top = pq.top(); pq.pop();
            if (!clusters[top.u].active || !clusters[top.v].active) continue;
            if (active_count <= max_outputs_ && top.loss > stop_threshold_) break;

            clusters[top.u].bbox = merge_bbox(clusters[top.u].bbox, clusters[top.v].bbox);
            clusters[top.u].area_sum += clusters[top.v].area_sum;
            clusters[top.v].active = false;
            active_count--;

            for (int k = 0; k < cn; ++k) {
                if (k != top.u && clusters[k].active) {
                    float new_loss = calculate_loss(clusters[top.u], clusters[k], cfg);
                    if (new_loss < std::numeric_limits<float>::infinity()) 
                        pq.push({new_loss, std::min(top.u, k), std::max(top.u, k)});
                }
            }
        }

        // 4. 汇总
        std::vector<BBox> final_results = direct_outputs;
        for (const auto& c : clusters) {
            if (c.active) final_results.push_back(finalize_box(c.bbox, img_w, img_h, cfg));
        }
        return final_results;
    }

private:
    int max_crop_size_, padding_, max_outputs_;
    float stop_threshold_;

    inline BBox merge_bbox(const BBox& a, const BBox& b) const {
        return BBox(std::min(a.x1, b.x1), std::min(a.y1, b.y1), std::max(a.x2, b.x2), std::max(a.y2, b.y2));
    }

    BBox finalize_box(BBox b, int img_w, int img_h, const Config& cfg) const {
        float w = b.width() + 2 * padding_;
        float h = b.height() + 2 * padding_;
        float cx = b.centerX();
        float cy = b.centerY();

        if (cfg.enable_aspect_ratio_fix) {
            float current_ar = w / h;
            if (current_ar < cfg.target_aspect_ratio) w = h * cfg.target_aspect_ratio;
            else h = w / cfg.target_aspect_ratio;
        }

        if (w > max_crop_size_) { 
            w = (float)max_crop_size_; 
            if(cfg.enable_aspect_ratio_fix) h = w / cfg.target_aspect_ratio; 
        }
        if (h > max_crop_size_) { 
            h = (float)max_crop_size_; 
            if(cfg.enable_aspect_ratio_fix) w = h * cfg.target_aspect_ratio; 
        }

        float x1 = std::max(0.0f, cx - w * 0.5f);
        float y1 = std::max(0.0f, cy - h * 0.5f);
        float x2 = std::min((float)img_w, x1 + w);
        float y2 = std::min((float)img_h, y1 + h);
        
        if (x2 == (float)img_w) x1 = std::max(0.0f, x2 - w);
        if (y2 == (float)img_h) y1 = std::max(0.0f, y2 - h);

        return BBox(x1, y1, x2, y2);
    }

    float calculate_loss(const Cluster& a, const Cluster& b, const Config& cfg) const {
        float x1 = std::min(a.bbox.x1, b.bbox.x1), y1 = std::min(a.bbox.y1, b.bbox.y1);
        float x2 = std::max(a.bbox.x2, b.bbox.x2), y2 = std::max(a.bbox.y2, b.bbox.y2);
        float rw = x2 - x1, rh = y2 - y1;

        if (rw + 2 * padding_ > max_crop_size_ || rh + 2 * padding_ > max_crop_size_) 
            return std::numeric_limits<float>::infinity();

        float max_dim = std::max(rw + 2 * padding_, rh + 2 * padding_);
        float l_size = std::exp(max_dim / (float)max_crop_size_) - 1.0f;
        float l_density = 1.0f - ((a.area_sum + b.area_sum) / (rw * rh + 1e-6f));
        float l_scale = std::abs(a.bbox.height() - b.bbox.height()) / (std::max(a.bbox.height(), b.bbox.height()) + 1e-6f);
        float l_square = 1.0f - ((rw * rh) / (max_dim * max_dim + 1e-6f));
        
        float dx = std::abs(a.bbox.centerX() - b.bbox.centerX());
        float dy = std::abs(a.bbox.centerY() - b.bbox.centerY());
        float l_align = std::min(dx, dy) / (std::max(dx, dy) + 1e-6f);

        return cfg.w_size * l_size + cfg.w_density * l_density + cfg.w_scale * l_scale + 
               cfg.w_square * l_square + cfg.w_alignment * l_align;
    }
};

#endif