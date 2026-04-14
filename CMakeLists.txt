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

add_executable(patch_test patch_test.cpp)
target_link_libraries(patch_test Eigen3::Eigen ${MOAB_LIBRARY})
target_include_directories(patch_test PRIVATE ${MOAB_INCLUDE_DIR})

enable_testing()
add_test(NAME patch_test COMMAND patch_test)
