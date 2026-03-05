#ifndef ANTIALIAS_H
#define ANTIALIAS_H

#include <vector>
#include <utility>
#include <random>
#include <cstddef>

template<typename Float = float>
using SampleOffset = std::pair<Float, Float>; 

inline std::vector<SampleOffset<float>> jittered_samples(int samples_per_pixel,
                                                         unsigned int seed = 12345u)
{
    std::vector<SampleOffset<float>> offsets;
    offsets.reserve(samples_per_pixel);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f); // [0,1)

    for (int s = 0; s < samples_per_pixel; ++s) {
        float dx = uni(rng) - 0.5f;   // [-0.5, 0.5)
        float dy = uni(rng) - 0.5f;
        offsets.emplace_back(dx, dy);
    }
    return offsets;
}

// ---- Wang hash for sub-pixel jitter ----
#ifdef __CUDACC__
__device__ inline float wang_hash_float(unsigned int seed) {
#else
inline float wang_hash_float(unsigned int seed) {
#endif
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed ^= seed >> 4u;
    seed *= 0x27d4eb2du;
    seed ^= seed >> 15u;
    return (float)seed / (float)0xFFFFFFFFu;
}

static constexpr int SAMPLES_PER_BATCH = 32;

#endif // ANTIALIAS_H
