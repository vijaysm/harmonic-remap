cmake_minimum_required(VERSION 3.10)
project(MimeticPatchTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(MOAB_PREFIX_HINTS
  /opt/moab/cmake-all
  /opt/moab/edgemaps
  /opt/moab/tempest
)

find_path(MOAB_INCLUDE_DIR
  moab/Core.hpp
  HINTS ${MOAB_PREFIX_HINTS}
  PATH_SUFFIXES include
)
find_library(MOAB_LIBRARY
  MOAB
  HINTS ${MOAB_PREFIX_HINTS}
  PATH_SUFFIXES lib
)

if(NOT MOAB_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find MOAB include directory containing moab/Core.hpp")
endif()

if(NOT MOAB_LIBRARY)
  message(FATAL_ERROR "Could not find MOAB library")
endif()

add_library(mimetic STATIC src/mimetic.cpp)
target_link_libraries(mimetic PUBLIC Eigen3::Eigen ${MOAB_LIBRARY})
target_include_directories(mimetic
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MOAB_INCLUDE_DIR}
)

add_executable(patch_test tests/patch_test.cpp)
target_link_libraries(patch_test PRIVATE mimetic)

add_executable(conservative_intersection_test tests/conservative_intersection_test.cpp)
target_link_libraries(conservative_intersection_test PRIVATE mimetic)

add_executable(voronoi_intersection_test tests/voronoi_intersection_test.cpp)
target_link_libraries(voronoi_intersection_test PRIVATE mimetic)

add_executable(convergence_validation_test tests/convergence_validation_test.cpp)
target_link_libraries(convergence_validation_test PRIVATE mimetic)

enable_testing()
add_test(NAME patch_test COMMAND patch_test)
add_test(NAME conservative_intersection_test COMMAND conservative_intersection_test)
add_test(NAME voronoi_intersection_test COMMAND voronoi_intersection_test)
add_test(NAME convergence_validation_test COMMAND convergence_validation_test)
