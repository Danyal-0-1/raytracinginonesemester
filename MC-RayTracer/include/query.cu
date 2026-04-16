// include/query.cu
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
       const BVHNode* __restrict__ nodes,
       const AABB* __restrict__ aabbs,
       const Triangle* __restrict__ triangles,
       const int32_t* __restrict__ triObjectIds,
       const Material* __restrict__ objectMaterials,
       const int numObjectMaterials,
       const Light* __restrict__ lights,
       const int numLights,
       const bool diffuse_bounce,
       const EmissiveTriInfo* __restrict__ emissiveTris,
       const float* __restrict__ emissiveCDF,
       const int numEmissiveTris,
       const float totalEmissiveArea,
       Vec3* __restrict__ output,
       Vec3* __restrict__ albedo_aov,
       Vec3* __restrict__ normal_aov,
       int nee_mode,
       const HomogeneousMedium* __restrict__ objectMedia,
       int numObjectMedia,
       const TextureData* __restrict__ textures,
       int numTextures,
       const VolumeRegionGPU* __restrict__ volumeRegions,
       int numVolumeRegions,
       const HDRTextureData* __restrict__ hdri)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int pix_id = y * W + x;
    const unsigned int pixel_seed = (unsigned int)x * 73856093u ^ (unsigned int)y * 19349663u;

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
            numTriangles,
            nodes, aabbs, triangles,
            triObjectIds, objectMaterials, numObjectMaterials,
            lights, numLights,
            emissiveTris, emissiveCDF, numEmissiveTris, totalEmissiveArea,
            rng,
            diffuse_bounce,
            nee_mode,
            objectMedia, numObjectMedia,
            textures, numTextures,
            volumeRegions, numVolumeRegions,
            hdri
        );
        batch_accum = batch_accum + color;

        // Write AOV buffers on the very first sample
        if (s == 0 && albedo_aov != nullptr && normal_aov != nullptr) {
            HitRecord aovHit;
            SearchBVH(numTriangles, ray, nodes, aabbs, triangles, aovHit);
            if (aovHit.hit) {
                assignMaterialToHit(aovHit, numTriangles, triObjectIds,
                                    objectMaterials, numObjectMaterials,
                                    textures, numTextures);
                albedo_aov[pix_id] = aovHit.mat.albedo;
                normal_aov[pix_id] = normalize(aovHit.normal);
            } else {
                Vec3 sky = (hdri && hdri->width > 0)
                           ? sampleHDRI(*hdri, ray.direction())
                           : missColor;
                albedo_aov[pix_id] = sky;
                normal_aov[pix_id] = make_vec3(0.0f, 0.0f, 0.0f);
            }
        }
    }

    output[pix_id] = output[pix_id] + batch_accum;
}

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
    const EmissiveTriInfo* __restrict__ emissiveTris,
    const float* __restrict__ emissiveCDF,
    const int numEmissiveTris,
    const float totalEmissiveArea,
    Vec3* __restrict__ output,
    Vec3* __restrict__ albedo_aov,
    Vec3* __restrict__ normal_aov,
    int nee_mode,
    const HomogeneousMedium* __restrict__ objectMedia,
    int numObjectMedia,
    const TextureData* __restrict__ textures,
    int numTextures,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions,
    const HDRTextureData* __restrict__ hdri)
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
            nodes,
            aabbs,
            triangles,
            triObjectIds,
            objectMaterials,
            numObjectMaterials,
            lights,
            numLights,
            diffuse_bounce,
            emissiveTris,
            emissiveCDF,
            numEmissiveTris,
            totalEmissiveArea,
            output,
            albedo_aov,
            normal_aov,
            nee_mode,
            objectMedia, numObjectMedia,
            textures, numTextures,
            volumeRegions, numVolumeRegions,
            hdri
        );
    }

    normalizeCUDA<<<tile_grid, block>>>(W, H, spp, output);
    CHECK_CUDA((cudaDeviceSynchronize()), true);

#else
    if (nodes == nullptr || aabbs == nullptr || triangles == nullptr || output == nullptr)
        return;

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
                    triCount,
                    nodes, aabbs, triangles,
                    triObjectIds, objectMaterials, numObjectMaterials,
                    lights, numLights,
                    emissiveTris, emissiveCDF, numEmissiveTris, totalEmissiveArea,
                    rng,
                    diffuse_bounce,
                    nee_mode,
                    objectMedia, numObjectMedia,
                    textures, numTextures,
                    volumeRegions, numVolumeRegions,
                    hdri
                );

                // Write AOVs on first sample
                if (si == 0 && albedo_aov != nullptr && normal_aov != nullptr) {
                    HitRecord aovHit;
                    SearchBVH(triCount, ray, nodes, aabbs, triangles, aovHit);
                    if (aovHit.hit) {
                        assignMaterialToHit(aovHit, triCount, triObjectIds,
                                            objectMaterials, numObjectMaterials,
                                            textures, numTextures);
                        albedo_aov[pix_id] = aovHit.mat.albedo;
                        normal_aov[pix_id] = normalize(aovHit.normal);
                    } else {
                        albedo_aov[pix_id] = missColor;
                        normal_aov[pix_id] = make_vec3(0.0f, 0.0f, 0.0f);
                    }
                }
            }
            output[pix_id] = col / float(spp);
        }
    }
#endif
}
