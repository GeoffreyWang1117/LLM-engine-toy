cmake_minimum_required(VERSION 3.10)
project(LLMEngineToy VERSION 0.1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 收集源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")

# 创建库
add_library(llm_engine STATIC ${SOURCES})

# 添加示例程序
add_executable(simple_example examples/simple_example.cpp)
target_link_libraries(simple_example llm_engine)

# 打印配置信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
