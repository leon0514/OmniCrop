#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "OmniCrop.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_omnicrop_backend, m) {
    py::class_<BBox>(m, "BBox")
        .def(py::init<float, float, float, float>())
        .def_readwrite("x1", &BBox::x1)
        .def_readwrite("y1", &BBox::y1)
        .def_readwrite("x2", &BBox::x2)
        .def_readwrite("y2", &BBox::y2)
        .def_property_readonly("width", &BBox::width)
        .def_property_readonly("height", &BBox::height);

    py::class_<OmniCropEngine::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("w_size", &OmniCropEngine::Config::w_size)
        .def_readwrite("w_density", &OmniCropEngine::Config::w_density)
        .def_readwrite("w_scale", &OmniCropEngine::Config::w_scale)
        .def_readwrite("w_square", &OmniCropEngine::Config::w_square)
        .def_readwrite("w_alignment", &OmniCropEngine::Config::w_alignment)
        .def_readwrite("min_obj_size", &OmniCropEngine::Config::min_obj_size)
        .def_readwrite("max_obj_size", &OmniCropEngine::Config::max_obj_size)
        .def_readwrite("enable_aspect_ratio_fix", &OmniCropEngine::Config::enable_aspect_ratio_fix)
        .def_readwrite("target_aspect_ratio", &OmniCropEngine::Config::target_aspect_ratio);

    py::class_<OmniCropEngine>(m, "OmniCropEngine")
        .def(py::init<int, int, int, float>(), 
             py::arg("max_crop_size") = 1280, 
             py::arg("padding") = 50, 
             py::arg("max_outputs") = 5, 
             py::arg("stop_threshold") = 3.5f)
        .def("cluster_and_crop", &OmniCropEngine::cluster_and_crop,
             py::arg("person_boxes"), 
             py::arg("img_w"), 
             py::arg("img_h"), 
             py::arg("cfg") = OmniCropEngine::Config());
}