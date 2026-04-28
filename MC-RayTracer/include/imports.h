#ifndef IMPORTS_CUH
#define IMPORTS_CUH

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <cstddef>
#include <random>
#include <cmath>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>

    #include <thrust/random.h>
    #include <thrust/iterator/counting_iterator.h>
    #include <thrust/functional.h>
    #include <thrust/remove.h>
    #include <thrust/gather.h>
    #include <thrust/unique.h>
    #include <thrust/host_vector.h>
    #include <thrust/device_vector.h>
    #include <thrust/reduce.h>
    #include <thrust/sort.h>
    #include <thrust/complex.h>
    #include <thrust/fill.h>
    #include <thrust/fill.h>
    #include <cooperative_groups.h>
    #include <cooperative_groups/reduce.h>
    
    namespace cg = cooperative_groups;

    #define HYBRID_FUNC __host__ __device__

    #define CHECK_CUDA(A, debug) \
    A; if(debug) { \
    auto ret = cudaDeviceSynchronize(); \
    if (ret != cudaSuccess) { \
    std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
    throw std::runtime_error(cudaGetErrorString(ret)); \
    } \
    }

#else
    #define HYBRID_FUNC
#endif

#ifndef __CUDACC__
HYBRID_FUNC inline float rsqrtf(float x) {
    return 1.0f / std::sqrt(x);
}
#endif


#ifndef __CUDACC__
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
    // force glm to work with cuda
#define GLM_FORCE_CUDA
#endif



#endif // IMPORTS_CUH
