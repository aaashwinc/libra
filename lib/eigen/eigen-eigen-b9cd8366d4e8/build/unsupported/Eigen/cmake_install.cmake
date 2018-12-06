# Install script for directory: /home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/AdolcForward"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/AlignedVector3"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/ArpackSupport"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/AutoDiff"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/BVH"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/FFT"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/IterativeSolvers"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/KroneckerProduct"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/LevenbergMarquardt"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/MatrixFunctions"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/MoreVectorization"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/MPRealSupport"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/NonLinearOptimization"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/NumericalDiff"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/OpenGLSupport"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/Polynomials"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/Skyline"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/SparseExtra"
    "/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/unsupported/Eigen/Splines"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/ashwin/repo/libra/lib/eigen/eigen-eigen-b9cd8366d4e8/build/unsupported/Eigen/src/cmake_install.cmake")

endif()

