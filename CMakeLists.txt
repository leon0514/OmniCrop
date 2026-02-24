cmake_minimum_required(VERSION 3.15)
project(OmniCrop)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 自动处理 pybind11 依赖
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    endif()
    find_package(pybind11 REQUIRED)
endif()

# 注意：这里直接生成目标，不要手动加路径前缀
pybind11_add_module(_omnicrop_backend src/bindings.cpp)
target_include_directories(_omnicrop_backend PRIVATE src)

# --- development-friendly output ------------------------------------------------
# when building locally (e.g. during `pip install -e .`), we want the shared
# object to live inside the Python package slice so that `import omnicrop`
# always finds it.  Set the library output directory accordingly.
set_target_properties(_omnicrop_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/omnicrop"
)

# [修正] Scikit-build-core 默认会将 install 的 target 放在 wheel 的根目录或指定目录
# 这里我们直接安装到包内部，这与上面的输出目录一致
install(TARGETS _omnicrop_backend DESTINATION omnicrop)