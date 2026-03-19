cmake_minimum_required(VERSION 3.18)
project(torch_fl CXX C)

set(FL_BACKEND "" CACHE STRING "Backend: npu | mlu (CUDA does not use CMake)")

# 公共框架源文件
file(GLOB_RECURSE COMMON_SRCS fl/csrc/**/*.cpp)

# 后端特有源文件
file(GLOB_RECURSE BACKEND_SRCS backends/${FL_BACKEND}/csrc/**/*.cpp)

# SDK 配置
if(FL_BACKEND STREQUAL "npu")
    set(SDK_HOME $ENV{ASCEND_HOME_PATH})
    set(SDK_LIBS ascendcl hccl)
    add_subdirectory(backends/npu/third_party/op-plugin)
elseif(FL_BACKEND STREQUAL "mlu")
    set(SDK_HOME $ENV{NEUWARE_HOME})
    set(SDK_LIBS cnnl cnrt cncl cndrv)
    if(USE_BANG)
        enable_language(BANG)
        file(GLOB_RECURSE BANG_SRCS backends/mlu/csrc/aten/operators/bang/**/*.mlu)
    endif()
endif()

# Python 扩展库
add_library(_C SHARED ${COMMON_SRCS} ${BACKEND_SRCS})
target_include_directories(_C PRIVATE
    fl/csrc
    backends/${FL_BACKEND}/csrc
    ${SDK_HOME}/include
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(_C ${SDK_LIBS} torch torch_python c10)
