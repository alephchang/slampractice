
cmake_minimum_required(VERSION 2.8)
project(L5)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package(Pangolin REQUIRED)
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
)

add_executable(computeORB computeORB.cpp)
target_link_libraries(computeORB ${OpenCV_LIBS} )

add_executable(E2Rt E2Rt.cpp)
target_link_libraries(E2Rt
/usr/local/lib/libSophus.so
)

add_executable(GN-BA GN-BA.cpp)
target_link_libraries(GN-BA
/usr/local/lib/libSophus.so
)

add_executable(icp icp.cpp)
target_link_libraries(icp
${Pangolin_LIBRARIES}
/usr/local/lib/libSophus.so
g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension 
${CSPARSE_LIBRARY}
)

#add_executable(disparity disparity.cpp)
#target_link_libraries(disparity
#${OpenCV_LIBS}
#${Pangolin_LIBRARIES}
#)

#add_executable(gaussnewton gaussnewton.cpp)
#target_link_libraries(gaussnewton
#${OpenCV_LIBS}
#)

