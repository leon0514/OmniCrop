#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "OmniCrop.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_omnicrop_backend, m)
{
    // 绑定 BBox 类
    py::class_<omnicrop::BBox>(m, "BBox")
        .def(py::init<float, float, float, float>())
        .def_readwrite("x1", &omnicrop::BBox::x1)
        .def_readwrite("y1", &omnicrop::BBox::y1)
        .def_readwrite("x2", &omnicrop::BBox::x2)
        .def_readwrite("y2", &omnicrop::BBox::y2)
        .def_property_readonly("width", &omnicrop::BBox::width)
        .def_property_readonly("height", &omnicrop::BBox::height);

    // 绑定 Config 结构体
    py::class_<omnicrop::Config>(m, "Config")
        .def(py::init<>())
        // 核心Affinity权重（用于决定合并的顺序）
        .def_readwrite("w_diou", &omnicrop::Config::w_diou)
        .def_readwrite("w_expansion", &omnicrop::Config::w_expansion)

        // 迭代优化评分参数（取代了 stop_threshold）
        // 该值越大，引擎越倾向于减少 Crop 数量（即合并更多目标）
        .def_readwrite("crop_count_penalty", &omnicrop::Config::crop_count_penalty)

        // 后处理参数
        .def_readwrite("nms_threshold", &omnicrop::Config::nms_threshold)

        // 画面适配参数
        .def_readwrite("enable_aspect_ratio_fix", &omnicrop::Config::enable_aspect_ratio_fix)
        .def_readwrite("target_aspect_ratio", &omnicrop::Config::target_aspect_ratio);

    // 绑定 OmniCropEngine 类
    py::class_<omnicrop::OmniCropEngine>(m, "OmniCropEngine")
        .def(py::init<int, int>(),
             py::arg("max_crop_size") = 1280,
             py::arg("padding") = 30)
        .def("cluster_and_crop", &omnicrop::OmniCropEngine::cluster_and_crop,
             py::arg("person_boxes"),
             py::arg("img_w"),
             py::arg("img_h"),
             py::arg("cfg") = omnicrop::Config());
}