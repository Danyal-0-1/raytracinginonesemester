cmake_minimum_required(VERSION 3.10)
project(RayTracerHW2)

# Build option: Defaults to CPU (OFF)
option(ENABLE_GPU "Enable CUDA implementation" OFF)

# Light system source files (CPU-only, no CUDA)
set(LIGHT_SOURCES
    src/EnvironmentLight.cpp
    src/LightFactory.cpp
    src/LightJsonParser.cpp
    src/EnvironmentLightRegistration.cpp
)

# Define source files
set(SOURCES
    src/main.cu
    include/bvh.cu
    include/query.cu
    ${LIGHT_SOURCES}
)

# Include directories
# Matches: -I ../third_party/glm/ 
include_directories(${CMAKE_SOURCE_DIR}/third_party/glm)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

if(ENABLE_GPU)
    message(STATUS "Building for GPU (CUDA Enabled)")

    enable_language(CUDA)

    # OptiX headers (bundled for denoiser support — no SDK install needed)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/optix)

    add_executable(render ${SOURCES})

    # Add CUDA specific flags (extended-lambda, relaxed-constexpr, fast-math)
    # --allow-unsupported-compiler: CUDA 11.8 doesn't officially support GCC 13+
    target_compile_options(render PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr;--use_fast_math;--allow-unsupported-compiler>)

    # Link CUDA driver API (needed by OptiX for optixInit) and C++ stdlib
    target_link_libraries(render PRIVATE cuda stdc++ m)

    # Ensure C++11 or higher (CUDA usually supports C++14/17 in recent versions)
    set_target_properties(render PROPERTIES CUDA_STANDARD 17)

else()
    message(STATUS "Building for CPU (Default)")
    
    enable_language(CXX)
    
    # Tell CMake to treat .cu files as C++ sources
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
    
    add_executable(render ${SOURCES})
    
    # Force GCC/Clang to treat .cu files as C++
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(render PRIVATE -x c++)
    endif()
    
    set_target_properties(render PROPERTIES CXX_STANDARD 17)
endif()
