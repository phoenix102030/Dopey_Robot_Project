cmake_minimum_required(VERSION 3.21)
project(robp_boot_camp_motors)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robp_interfaces REQUIRED)

find_package(Boost REQUIRED random)

include_directories(include)

add_custom_target(
  copy_kobuki_motors_lib ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/lib/libkobuki_motors.so ${CMAKE_CURRENT_BINARY_DIR}/libkobuki_motors.so
)

add_library(kobuki_motors SHARED IMPORTED)
set_target_properties(kobuki_motors PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(kobuki_motors PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libkobuki_motors.so)
add_dependencies(kobuki_motors copy_kobuki_motors_lib)

add_executable(motors src/motors.cpp)
target_link_libraries(motors Boost::random kobuki_motors)
ament_target_dependencies(motors geometry_msgs rclcpp robp_interfaces)

install(TARGETS
  motors
	DESTINATION lib/${PROJECT_NAME}
)

install(IMPORTED_RUNTIME_ARTIFACTS
  kobuki_motors
	DESTINATION lib/${PROJECT_NAME}
)

ament_package()