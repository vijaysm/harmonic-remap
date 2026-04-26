cmake_minimum_required(VERSION 3.10)
project(MimeticPatchTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

option(MIMETIC_ENABLE_OPENMP "Enable OpenMP parallel acceleration" OFF)

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

find_path(NANOFLANN_INCLUDE_DIR nanoflann.hpp
  HINTS /opt/homebrew/include /usr/local/include)

add_library(mimetic STATIC src/mimetic.cpp)
target_link_libraries(mimetic PUBLIC Eigen3::Eigen ${MOAB_LIBRARY})
target_include_directories(mimetic
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MOAB_INCLUDE_DIR}
    ${NANOFLANN_INCLUDE_DIR}
)

if(MIMETIC_ENABLE_OPENMP)
  set(_openmp_configured OFF)
  if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_path(OPENMP_OMP_INCLUDE_DIR
      omp.h
      HINTS
        ${OpenMP_ROOT}/include
        /opt/homebrew/opt/libomp/include
        /opt/homebrew/Cellar/libomp/21.1.3/include
    )
    find_library(OPENMP_OMP_LIBRARY
      NAMES omp
      HINTS
        ${OpenMP_ROOT}/lib
        /opt/homebrew/opt/libomp/lib
        /opt/homebrew/Cellar/libomp/21.1.3/lib
    )
    if(OPENMP_OMP_INCLUDE_DIR AND OPENMP_OMP_LIBRARY)
      target_compile_options(mimetic PUBLIC -Xpreprocessor -fopenmp)
      target_include_directories(mimetic PUBLIC ${OPENMP_OMP_INCLUDE_DIR})
      target_link_libraries(mimetic PUBLIC ${OPENMP_OMP_LIBRARY})
      set(_openmp_configured ON)
    endif()
  endif()

  if(NOT _openmp_configured)
    find_package(OpenMP REQUIRED)
    target_link_libraries(mimetic PUBLIC OpenMP::OpenMP_CXX)
  endif()
  target_compile_definitions(mimetic PUBLIC MIMETIC_ENABLE_OPENMP=1)
endif()

add_executable(patch_test tests/patch_test.cpp)
target_link_libraries(patch_test PRIVATE mimetic)

add_executable(conservative_intersection_test tests/conservative_intersection_test.cpp)
target_link_libraries(conservative_intersection_test PRIVATE mimetic)

add_executable(voronoi_intersection_test tests/voronoi_intersection_test.cpp)
target_link_libraries(voronoi_intersection_test PRIVATE mimetic)

add_executable(convergence_validation_test tests/convergence_validation_test.cpp)
target_link_libraries(convergence_validation_test PRIVATE mimetic)

add_executable(spherical_quad_test tests/spherical_quad_test.cpp)
target_link_libraries(spherical_quad_test PRIVATE mimetic)

add_executable(spherical_geometry_test tests/spherical_geometry_test.cpp)
target_link_libraries(spherical_geometry_test PRIVATE mimetic)

add_executable(spherical_voronoi_test tests/spherical_voronoi_test.cpp)
target_link_libraries(spherical_voronoi_test PRIVATE mimetic)

add_executable(spherical_scalar_test tests/spherical_scalar_test.cpp)
target_link_libraries(spherical_scalar_test PRIVATE mimetic)

add_executable(cell_average_transfer_test tests/cell_average_transfer_test.cpp)
target_link_libraries(cell_average_transfer_test PRIVATE mimetic)

add_executable(hdiv_conforming_projection_test tests/hdiv_conforming_projection_test.cpp)
target_link_libraries(hdiv_conforming_projection_test PRIVATE mimetic)

add_executable(high_order_edge_moment_test tests/high_order_edge_moment_test.cpp)
target_link_libraries(high_order_edge_moment_test PRIVATE mimetic)

add_executable(high_order_hdiv_convergence_test tests/high_order_hdiv_convergence_test.cpp)
target_link_libraries(high_order_hdiv_convergence_test PRIVATE mimetic)

add_executable(spherical_high_order_moment_test tests/spherical_high_order_moment_test.cpp)
target_link_libraries(spherical_high_order_moment_test PRIVATE mimetic)

add_executable(spherical_high_order_hdiv_convergence_test tests/spherical_high_order_hdiv_convergence_test.cpp)
target_link_libraries(spherical_high_order_hdiv_convergence_test PRIVATE mimetic)

enable_testing()
add_test(NAME patch_test COMMAND patch_test)
add_test(NAME conservative_intersection_test COMMAND conservative_intersection_test)
add_test(NAME voronoi_intersection_test COMMAND voronoi_intersection_test)
add_test(NAME convergence_validation_test COMMAND convergence_validation_test)
add_test(NAME spherical_geometry_test COMMAND spherical_geometry_test)
add_test(NAME spherical_quad_test COMMAND spherical_quad_test)
add_test(NAME spherical_voronoi_test COMMAND spherical_voronoi_test)
add_test(NAME spherical_scalar_test COMMAND spherical_scalar_test)
add_test(NAME cell_average_transfer_test COMMAND cell_average_transfer_test)
add_test(NAME hdiv_conforming_projection_test COMMAND hdiv_conforming_projection_test)
add_test(NAME high_order_edge_moment_test COMMAND high_order_edge_moment_test)
add_test(NAME high_order_hdiv_convergence_test COMMAND high_order_hdiv_convergence_test)
add_test(NAME spherical_high_order_moment_test COMMAND spherical_high_order_moment_test)
add_test(NAME spherical_high_order_hdiv_convergence_test COMMAND spherical_high_order_hdiv_convergence_test)

add_executable(dump_visuals tests/dump_visuals.cpp)
target_link_libraries(dump_visuals PRIVATE mimetic ${MOAB_LIBRARY})

add_executable(roundtrip_analysis tests/roundtrip_analysis.cpp)
target_link_libraries(roundtrip_analysis PRIVATE mimetic ${MOAB_LIBRARY})
target_include_directories(roundtrip_analysis PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests ${MOAB_INCLUDE_DIRS})
