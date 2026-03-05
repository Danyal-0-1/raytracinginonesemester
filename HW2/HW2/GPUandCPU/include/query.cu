#include "buffers.h"
#include "query.h"
#include "scene.h"
#include "shader.h"

#include <cfloat>
#include <cmath>


#ifdef __CUDACC__

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderBatchCUDA(const int numTriangles,
       int W, int H,
       int max_depth,
       int sample_begin,
       int sample_end,
       const Camera cam,
       const Vec3 missColor,
       const Environment env,
       const BVHNode* __restrict__ nodes,
       const AABB* __restrict__ aabbs,
       const Triangle* __restrict__ triangles,
       const int32_t* __restrict__ triObjectIds,
       const Material* __restrict__ objectMaterials,
       const int numObjectMaterials,
       const Light* __restrict__ lights,
       const int numLights,
       const bool diffuse_bounce,
       Vec3* __restrict__ output)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int pix_id = y * W + x;
    const unsigned int pixel_seed = (unsigned int)x * 73856093u ^ (unsigned int)y * 19349663u;

    // Accumulate this batch in registers — no global memory traffic per sample
    Vec3 batch_accum = make_vec3(0.0f, 0.0f, 0.0f);

    for (int s = sample_begin; s < sample_end; ++s) {
        unsigned int h = pixel_seed ^ (unsigned int)(s * 83492791u);
        float jx = wang_hash_float(h) - 0.5f;
        h = h * 1664525u + 1013904223u;
        float jy = wang_hash_float(h) - 0.5f;

        Ray ray = cam.get_ray((float)x + jx, (float)y + jy);

        unsigned int rng = make_rng_seed(x, y, s);
        Vec3 color = TraceRayIterative(
            ray,
            max_depth,
            missColor,
            env,
            numTriangles,
            nodes, aabbs, triangles,
            triObjectIds, objectMaterials, numObjectMaterials,
            lights, numLights,
            rng,
            diffuse_bounce
        );
        batch_accum = batch_accum + color;
    }

    // Single read-modify-write per batch instead of per sample
    output[pix_id] = output[pix_id] + batch_accum;
}

// Divide accumulated buffer by sample count.
__global__ void normalizeCUDA(int W, int H, int spp, Vec3* __restrict__ output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int pix_id = y * W + x;
    output[pix_id] = output[pix_id] / float(spp);
}

#endif

void render(
    const size_t numTriangles,
    int W, int H,
    const Camera cam,
    const Vec3 missColor,
    const Environment env,
    const int max_depth,
    const int spp,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    const int32_t* __restrict__ triObjectIds,
    const Material* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const Light* __restrict__ lights,
    const int numLights,
    const bool diffuse_bounce,
    Vec3* __restrict__ output)
{
#ifdef __CUDACC__
    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    for (int s = 0; s < spp; s += SAMPLES_PER_BATCH) {
        int batch_end = s + SAMPLES_PER_BATCH;
        if (batch_end > spp) batch_end = spp;

        renderBatchCUDA<<<tile_grid, block>>>(
            numTriangles,
            W, H,
            max_depth,
            s,
            batch_end,
            cam,
            missColor,
            env,
            nodes,
            aabbs,
            triangles,
            triObjectIds,
            objectMaterials,
            numObjectMaterials,
            lights,
            numLights,
            diffuse_bounce,
            output
        );
    }

    // Normalize: divide accumulated color by spp
    normalizeCUDA<<<tile_grid, block>>>(W, H, spp, output);
    CHECK_CUDA((cudaDeviceSynchronize()), true);

#else
    if (nodes == nullptr || aabbs == nullptr || triangles == nullptr || output == nullptr) {
        return;
    }

    const int triCount = static_cast<int>(numTriangles);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int pix_id = W * y + x;

            Vec3 col{0,0,0};

            auto offsets = jittered_samples(spp, 42u);

            for (int si = 0; si < (int)offsets.size(); ++si) {
                float px = float(x) + offsets[si].first;
                float py = float(y) + offsets[si].second;

                const Ray ray = cam.get_ray(px, py);

                unsigned int rng = make_rng_seed(x, y, si);
                col = col + TraceRayIterative(
                    ray,
                    max_depth,
                    missColor,
                    env,
                    triCount,
                    nodes, aabbs, triangles,
                    triObjectIds, objectMaterials, numObjectMaterials,
                    lights, numLights,
                    rng,
                    diffuse_bounce
                );
            }
            output[pix_id] = col/float(spp);
        }
    }
#endif
}
