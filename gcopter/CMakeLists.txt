cmake_minimum_required(VERSION 2.8.3)

project(gcopter)

set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")

find_package(Eigen3 REQUIRED)
find_package(ompl REQUIRED)
find_package(PCL REQUIRED COMPONENTS common octree filters)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
)

add_subdirectory(include/corridor_gen)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OMPL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  include
)
link_directories(${PCL_LIBRARY_DIRS})
catkin_package()

add_executable(global_planning src/global_planning.cpp)

target_link_libraries(global_planning
  ${OMPL_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_executable(global_planning_bubble src/global_planning_bubble.cpp)

target_link_libraries(global_planning_bubble
  ${OMPL_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  corridor_gen
)
