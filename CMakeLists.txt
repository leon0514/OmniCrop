cmake_minimum_required(VERSION 3.15)
project(OmniCrop)

# 自动寻找 pybind11 (由 scikit-build-core 在构建时处理依赖)
find_package(pybind11 REQUIRED)

# 创建扩展模块
pybind11_add_module(_omnicrop_backend src/bindings.cpp)

# 如果你的 OmniCrop.hpp 在 src 目录下
target_include_directories(_omnicrop_backend PRIVATE src)

# 设置 C++ 标准
set_target_properties(_omnicrop_backend PROPERTIES CXX_STANDARD 11)

# 安装规则 (供 Python 寻找编译好的 .so 或 .pyd 文件)
install(TARGETS _omnicrop_backend DESTINATION OmniCrop)