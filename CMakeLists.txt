cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(softmax LANGUAGES C CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 语言标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 指定 CUDA 的编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
# sm_50 是 CUDA 架构版本，请根据你的 GPU 修改
# 例如：sm_60、sm_70 或 sm_75 
set(CMAKE_EXPORT_COMPILE_COMMANDS=1)
# 添加项目包含目录
include_directories(${CMAKE_SOURCE_DIR})

# 查找当前目录中的所有源文件，并将列表存储在 SRCS 变量中
file(GLOB SRCS ${CMAKE_SOURCE_DIR}/*.cu)

# 添加可执行文件并将 SRCS 传递给它
add_executable(main ${SRCS})

# 配置可执行文件链接的库
target_link_libraries(main ${CUDA_LIBRARIES})