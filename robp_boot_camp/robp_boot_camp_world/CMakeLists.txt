cmake_minimum_required(VERSION 3.5)
project(robp_boot_camp_world)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(angles REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

include_directories(include)

add_executable(world src/world.cpp)
ament_target_dependencies(world angles geometry_msgs rclcpp std_msgs std_srvs tf2_geometry_msgs tf2_ros visualization_msgs)

install(TARGETS
  world
	DESTINATION lib/${PROJECT_NAME})

ament_package()