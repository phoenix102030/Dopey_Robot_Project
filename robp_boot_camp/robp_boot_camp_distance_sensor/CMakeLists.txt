cmake_minimum_required(VERSION 3.21)
project(robp_boot_camp_distance_sensor)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robp_boot_camp_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(Boost REQUIRED random)


include_directories(include)

add_custom_target(
  copy_distance_sensor_lib ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/lib/libdistance_sensor.so ${CMAKE_CURRENT_BINARY_DIR}/libdistance_sensor.so
)

add_library(distance_sensor SHARED IMPORTED)
set_target_properties(distance_sensor PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(distance_sensor PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libdistance_sensor.so)
add_dependencies(distance_sensor copy_distance_sensor_lib)

add_executable(distance_sensor_node src/distance_sensor_node.cpp)
target_link_libraries(distance_sensor_node Boost::random distance_sensor)
ament_target_dependencies(distance_sensor_node geometry_msgs rclcpp robp_boot_camp_interfaces std_msgs tf2 tf2_geometry_msgs tf2_ros visualization_msgs)

install(TARGETS
distance_sensor_node
	DESTINATION lib/${PROJECT_NAME}
)

install(IMPORTED_RUNTIME_ARTIFACTS
distance_sensor
	DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()