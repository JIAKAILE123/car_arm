cmake_minimum_required(VERSION 3.8)
project(yzbot_gazebo_plugins)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
add_library(dyn_obs_mover SHARED src/dyn_obs_mover.cpp)
target_include_directories(dyn_obs_mover PUBLIC ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(dyn_obs_mover ${GAZEBO_LIBRARIES})
ament_target_dependencies(dyn_obs_mover gazebo_dev gazebo_ros)
set_target_properties(dyn_obs_mover PROPERTIES
  CXX_STANDARD 14
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS dyn_obs_mover
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

ament_environment_hooks("${ament_cmake_package_templates_ENVIRONMENT_HOOK_LIBRARY_PATH}")
ament_package()
