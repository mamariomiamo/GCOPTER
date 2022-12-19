cmake_minimum_required(VERSION 3.10)
project(graph_search)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++17")
find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  plan_env
  cv_bridge
)

catkin_package(
 INCLUDE_DIRS include
 CATKIN_DEPENDS plan_env
#  DEPENDS system_lib
)

add_library(${PROJECT_NAME} src/graph_search.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen
    ${catkin_LIBRARIES}
)