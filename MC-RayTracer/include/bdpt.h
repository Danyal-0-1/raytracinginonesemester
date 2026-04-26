// Bidirectional Path Tracing (BDPT) — Mitsuba/PBRT-style, single-megakernel layout.
//
// Reference: Veach 1997 Ch. 10; PBRT-v3 src/integrators/bdpt.cpp; Mitsuba 3
// src/render/path.cpp; Pauly/Kollig/Keller 2000 for volumetric BDPT.

#pragma once

#include "camera.h"
#include "ray.h"
#include "MeshOBJ.h"
#include "brdf.h"
#include "shader.h"
#include "bvh.h"
#include "antialias.h"
#include "medium.h"
#include "texture.h"
#include "query.h"

#ifndef BDPT_MAX_DEPTH
#define BDPT_MAX_DEPTH 8
#endif

// Per-strategy luminance clamp (Mitsuba "max_clamp" / "splat clamp"). Bounds
// each (s,t) contribution before it's accumulated into L, suppressing the
// fireflies and dark dots that appear in dense participating media where
// ratio-tracked Tr / pdf ratios produce occasional huge spikes whose mean
// converges only at very high spp. Adds bias proportional to the clamp.
// Set to 0 or a negative value to disable.
#ifndef BDPT_FIREFLY_CLAMP
#define BDPT_FIREFLY_CLAMP 50.0f
#endif

HYBRID_FUNC inline Vec3 bdpt_clamp_firefly(const Vec3& C) {
    if (BDPT_FIREFLY_CLAMP <= 0.0f) return C;
    const float L = luminance(C);
    if (!(L > 0.0f) || L <= BDPT_FIREFLY_CLAMP) return C;   // also catches NaN via !(L>0)
    const float k = BDPT_FIREFLY_CLAMP / L;
    return make_vec3(C.x * k, C.y * k, C.z * k);
}

HYBRID_FUNC inline Vec3 bdpt_sanitize(const Vec3& v) {
    Vec3 r = v;
    if (!(r.x > -1e30f && r.x < 1e30f)) r.x = 0.0f;
    if (!(r.y > -1e30f && r.y < 1e30f)) r.y = 0.0f;
    if (!(r.z > -1e30f && r.z < 1e30f)) r.z = 0.0f;
    return r;
}

// ---------------------------------------------------------------------------
// PathVertex
// ---------------------------------------------------------------------------

enum BDPTVertexType : uint8_t {
    BDPT_VTX_CAMERA  = 0,
    BDPT_VTX_LIGHT   = 1,
    BDPT_VTX_SURFACE = 2,
    BDPT_VTX_MEDIUM  = 3,
};

struct PathVertex {
    BDPTVertexType type;
    bool   delta;          // true if scattering at this vertex is delta (pinhole camera, perfect specular)
    int    medium_id;      // index into volumeRegions[] for the medium *containing* this vertex; -1 = vacuum

    Vec3   p;              // world-space position
    Vec3   n;              // shading normal (zero for camera/medium); also used as emission lobe normal at lig[0]
    Vec3   ng;             // geometric normal (= n for now)
    Vec3   wo;             // direction from p back toward the previous vertex (unit). Zero on path endpoints.

    Vec3   beta;           // throughput at this vertex (PBRT alpha_E_t / alpha_L_s convention)

    float  pdf_fwd;        // area pdf of this vertex given previous vertex (in ORIGINAL walk direction)
    float  pdf_rev;        // area pdf of this vertex given next vertex (filled lazily during MIS)

    HitRecord hit;         // surface scattering data (BSDF/material/uvs); valid for SURFACE
    Vec3   Le;             // emitted radiance — nonzero for emitter endpoints and emissive surfaces hit by camera path

    // Step 2 placeholders:
    Vec3   sigma_s;
    Vec3   sigma_t;
    float  phase_g;
};

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

HYBRID_FUNC inline float convert_density_to_area(
    float pdf_solid_angle,
    const PathVertex& cur,
    const PathVertex& next)
{
    Vec3 d = next.p - cur.p;
    float r2 = dot(d, d);
    if (r2 < 1e-20f) return 0.0f;
    float inv_r2 = 1.0f / r2;
    float pdf = pdf_solid_angle * inv_r2;
    if (next.type == BDPT_VTX_SURFACE || next.type == BDPT_VTX_LIGHT) {
        Vec3 wn = d * sqrtf(inv_r2);
        pdf *= fabsf(dot(next.n, wn));
    }
    return pdf;
}

HYBRID_FUNC inline float geometry_term(const PathVertex& a, const PathVertex& b) {
    Vec3 d = b.p - a.p;
    float r2 = dot(d, d);
    if (r2 < 1e-20f) return 0.0f;
    float inv_r = rsqrtf(r2);
    Vec3 w = d * inv_r;
    float G = 1.0f / r2;
    if (a.type == BDPT_VTX_SURFACE || a.type == BDPT_VTX_LIGHT) G *= fabsf(dot(a.n,  w));
    if (b.type == BDPT_VTX_SURFACE || b.type == BDPT_VTX_LIGHT) G *= fabsf(dot(b.n, -w));
    return G;
}

// True if this hit is a delta-BSDF surface: perfect mirror (kr) OR dielectric (ior>1).
HYBRID_FUNC inline bool is_delta_surface(const HitRecord& hit) {
    if (hit.mat.kd > 1e-6f || hit.mat.ks > 1e-6f) return false;
    return (hit.mat.kr > 1e-6f) || (hit.mat.ior > 1.0f);
}

// True if material is a dielectric (ior > 1, no diffuse/glossy lobes). Matches
// the PT integrator's branch priority: ior>1 wins over kr (so a sphere with
// kr=1 AND ior=1.5 is interpreted as glass, not as a perfect mirror).
HYBRID_FUNC inline bool is_dielectric(const HitRecord& hit) {
    return hit.mat.kd <= 1e-6f && hit.mat.ks <= 1e-6f && hit.mat.ior > 1.0f;
}

// BSDF / phase value at vertex v. Returns f(wo_view, wi_light), no cosθ folded in.
//   wi = direction at v pointing toward "next" / light side
//   wo = direction at v pointing toward "previous" / camera side
// Delta vertices return zero (their contribution is captured by the random walk
// itself, never by an explicit connection — connections at deltas have measure zero).
HYBRID_FUNC inline Vec3 vertex_eval_f(const PathVertex& v, const Vec3& wi, const Vec3& wo) {
    if (v.type == BDPT_VTX_SURFACE) {
        if (v.delta) return make_vec3(0,0,0);
        return EvaluateBRDF(v.hit, wo, wi);
    }
    if (v.type == BDPT_VTX_MEDIUM) {
        const float cos_t = dot(normalize(wo), normalize(wi));
        const float g = v.phase_g;
        const float denom = 1.0f + g*g - 2.0f*g*cos_t;
        const float p = (1.0f - g*g) /
            (4.0f * 3.14159265358979323846f * denom * sqrtf(fmaxf(denom, 1e-20f)));
        return make_vec3(p, p, p);
    }
    return make_vec3(0.0f, 0.0f, 0.0f);
}

// Solid-angle pdf of sampling direction wi at v given incoming view wo.
HYBRID_FUNC inline float vertex_eval_pdf_dir(const PathVertex& v, const Vec3& wi, const Vec3& wo) {
    if (v.type == BDPT_VTX_SURFACE) return BRDFpdf(v.hit, wo, wi);
    if (v.type == BDPT_VTX_MEDIUM) {
        const float cos_t = dot(normalize(wo), normalize(wi));
        const float g = v.phase_g;
        const float denom = 1.0f + g*g - 2.0f*g*cos_t;
        return (1.0f - g*g) /
            (4.0f * 3.14159265358979323846f * denom * sqrtf(fmaxf(denom, 1e-20f)));
    }
    return 0.0f;
}

// Closed-form distance pdf for sampling a HOMOGENEOUS medium vertex `next` from
// origin `from_pos`. The forward-walk distance pdf for an inverse-CDF free-flight
// sampler with channel-uniform spectral handling is mean(sigma_t * exp(-sigma_t * t))
// where t is the geometric segment length through the medium. Returns 1.0 when the
// next vertex is non-medium or sits in a heterogeneous (density-grid) region —
// in those cases we fall back to the directional-only pdf approximation, matching
// PBRT's volumetric BDPT. Used symmetrically by the forward walk and by vertex_pdf
// during MIS reverse-pdf evaluation.
HYBRID_FUNC inline float homogeneous_medium_dist_pdf(
    const Vec3& from_pos, const PathVertex& next,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions)
{
    if (next.type != BDPT_VTX_MEDIUM) return 1.0f;
    if (next.medium_id < 0 || next.medium_id >= numVolumeRegions) return 1.0f;
    if (volumeRegions == nullptr) return 1.0f;
    const VolumeRegionGPU& vol = volumeRegions[next.medium_id];
    if (vol.has_density_grid()) return 1.0f;        // heterogeneous: PBRT-style approximation
    if (!vol.medium.enabled || !vol.medium.has_extinction()) return 1.0f;

    Vec3 d = next.p - from_pos;
    float r2 = dot(d, d);
    if (r2 < 1e-12f) return 0.0f;
    float dist = sqrtf(r2);

    Vec3 dir = d * (1.0f / dist);
    Ray r(from_pos, dir);
    float t_enter, t_exit;
    float seg_in;
    if (vol.ray_interval(r, t_enter, t_exit)) {
        const float seg_start = fmaxf(0.0f, t_enter);
        const float seg_end   = fminf(dist, t_exit);
        seg_in = fmaxf(0.0f, seg_end - seg_start);
    } else {
        seg_in = 0.0f;
    }
    if (seg_in <= 0.0f) return 0.0f;

    Vec3 Tr = vol.medium.transmittance(seg_in);
    Vec3 density = vol.medium.sigma_t * Tr;
    return spectrum_average(density);
}

// PBRT-style Vertex::Pdf: area- (or volume-) measure pdf of sampling 'next' from
// 'v' given came from 'prev'. 'prev' may be null when v is the first non-endpoint
// vertex on its subpath. When `next` is a homogeneous medium vertex, the result
// includes the analytic distance pdf so the MIS recursive ratio distinguishes
// strategies by expected medium traversal length. Heterogeneous medium vertices
// use the directional-only PBRT approximation.
HYBRID_FUNC inline float vertex_pdf(const PathVertex& v, const PathVertex* prev, const PathVertex& next,
                                    const VolumeRegionGPU* __restrict__ volumeRegions = nullptr,
                                    int numVolumeRegions = 0) {
    float pdf_area;
    if (v.type == BDPT_VTX_LIGHT) {
        Vec3 wn = next.p - v.p; float r2 = dot(wn, wn);
        if (r2 < 1e-20f) return 0.0f;
        wn = wn * rsqrtf(r2);
        float cos_t = fmaxf(dot(v.n, wn), 0.0f);
        float pdf_dir = cos_t * 0.31830988618379067f; // 1/π
        pdf_area = convert_density_to_area(pdf_dir, v, next);
    } else if (v.type == BDPT_VTX_CAMERA) {
        pdf_area = convert_density_to_area(1.0f, v, next);
    } else {
        Vec3 wn = next.p - v.p; float r2 = dot(wn, wn);
        if (r2 < 1e-20f) return 0.0f;
        wn = wn * rsqrtf(r2);
        Vec3 wp = (prev != nullptr) ? (prev->p - v.p) : make_vec3(0,0,0);
        float lp = sqrtf(fmaxf(dot(wp, wp), 1e-30f));
        if (prev) wp = wp * (1.0f / lp); else wp = v.wo;
        float pdf_dir = vertex_eval_pdf_dir(v, wn, wp);
        pdf_area = convert_density_to_area(pdf_dir, v, next);
    }
    // Add the medium-vertex distance pdf factor for homogeneous media (1.0 elsewhere).
    pdf_area *= homogeneous_medium_dist_pdf(v.p, next, volumeRegions, numVolumeRegions);
    return pdf_area;
}

// Area pdf of sampling the light vertex's POSITION on its emitter (independent of direction).
HYBRID_FUNC inline float vertex_pdf_light_origin(float totalEmissiveArea) {
    return (totalEmissiveArea > 0.0f) ? (1.0f / totalEmissiveArea) : 0.0f;
}

// ---------------------------------------------------------------------------
// Transmittance along an unobstructed connection segment a -> b.
// Homogeneous regions: closed-form exp(-sigma_t * len).
// Heterogeneous regions (density grid): ratio tracking via estimateTransmittance.
// ---------------------------------------------------------------------------
HYBRID_FUNC inline Vec3 segment_transmittance(
    const Vec3& a, const Vec3& b,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions,
    unsigned int& rng_state)
{
    Vec3 d = b - a;
    float dist2 = dot(d, d);
    if (dist2 < 1e-12f || numVolumeRegions == 0 || volumeRegions == nullptr)
        return make_vec3(1.0f, 1.0f, 1.0f);
    float dist = sqrtf(dist2);
    Vec3 dir = d * (1.0f / dist);
    Ray r(a, dir);

    Vec3 Tr = make_vec3(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < numVolumeRegions; ++i) {
        const VolumeRegionGPU& vol = volumeRegions[i];
        if (!vol.medium.enabled || !vol.medium.has_extinction()) continue;
        float t_enter, t_exit;
        if (!vol.ray_interval(r, t_enter, t_exit)) continue;
        const float seg_start = fmaxf(0.0f, t_enter);
        const float seg_end   = fminf(dist, t_exit);
        const float seg_len   = seg_end - seg_start;
        if (seg_len <= 0.0f) continue;

        if (vol.has_density_grid()) {
            // Ratio tracking expects a ray STARTING at the volume entry.
            Ray sub(r.at(seg_start), dir);
            Tr = Tr * estimateTransmittance(sub, seg_len, &vol, vol.medium, rng_state);
        } else {
            Tr = Tr * vol.medium.transmittance(seg_len);
        }
    }
    return Tr;
}

// ---------------------------------------------------------------------------
// Shared random walk: extends path[n_start..maxDepth] by sampling surface
// scattering and (homogeneous) medium scattering. Used by both camera and
// light subpaths — they only differ in how the endpoint vertex is built.
//
// Conventions — caller has already filled path[n_start - 1] (the endpoint)
// and computed throughput / pdf_dir for the FIRST step (i.e. for sampling
// path[n_start]). The walk fills path[n_start], path[n_start + 1], ... and
// returns the new vertex count.
//
// Step 2 scope: surfaces + homogeneous-medium scatter. Heterogeneous volumes
// are passed through (skipped — step 3).
// ---------------------------------------------------------------------------
HYBRID_FUNC inline int random_walk(
    Ray ray,
    Vec3 throughput,
    float pdf_dir,
    int maxDepth,
    int n_start,
    const int numTriangles,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    const int32_t* __restrict__ triObjectIds,
    const Material* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const TextureData* __restrict__ textures,
    int numTextures,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions,
    unsigned int& rng_state,
    PathVertex* path,
    // Optional output: accumulated camera-path contribution from emissive
    // (heterogeneous) media along the walk. Pass nullptr to skip (e.g. light walk).
    Vec3* out_vol_emission = nullptr)
{
    int n_vtx = n_start;
    int safety = 0;
    const int safety_max = 4 * maxDepth + 8;

    while (n_vtx <= maxDepth && safety++ < safety_max) {
        // ---------- Active medium at ray origin ----------
        int activeVolumeIdx = findVolumeIndexAtPoint(
            ray.origin(), volumeRegions, numVolumeRegions);
        HomogeneousMedium activeMedium = findVolumeAtPoint(
            ray.origin(), volumeRegions, numVolumeRegions);
        const VolumeRegionGPU* activeVolume =
            (activeVolumeIdx >= 0 && volumeRegions != nullptr)
            ? &volumeRegions[activeVolumeIdx] : nullptr;

        float t_volume_exit = 1e30f;
        if (activeVolume != nullptr) {
            float t_enter = 0.0f;
            if (!activeVolume->ray_interval(ray, t_enter, t_volume_exit)) {
                activeMedium.enabled = false;
                t_volume_exit = 1e30f;
                activeVolume = nullptr;
            }
        }

        // ---------- BVH for nearest surface ----------
        HitRecord hit;
        SearchBVH(numTriangles, ray, nodes, aabbs, triangles, hit);
        const float t_surf = hit.hit ? hit.t : 1e30f;

        // If we're outside any volume, see if a volume sits ahead before the
        // surface. If so, advance the ray to its entry and reloop.
        if (activeVolume == nullptr) {
            float t_volume_enter = 1e30f;
            float t_next_volume_exit_unused = 1e30f;
            int nextVolumeIdx = findNextVolumeAlongRay(
                ray, volumeRegions, numVolumeRegions,
                t_volume_enter, t_next_volume_exit_unused);
            if (nextVolumeIdx >= 0 && t_volume_enter + RT_EPS < t_surf) {
                ray = Ray(ray.at(t_volume_enter + RT_EPS), ray.direction());
                continue;
            }
        }

        // ---------- Volume scatter decision ----------
        if (activeVolume != nullptr && activeMedium.has_extinction()) {
            const float t_medium_limit = fminf(t_surf, t_volume_exit);
            if (t_medium_limit > RT_EPS) {
                const bool  heterogeneous = activeVolume->has_density_grid();
                const int   channel = (int)fminf(2.0f, floorf(rng_next(rng_state) * 3.0f));
                float t_vol = 0.0f;
                bool  sampled_scatter;
                Vec3 vol_emission_local = make_vec3(0,0,0);
                if (heterogeneous) {
                    // Capture volume emission only for the camera walk (out_vol_emission != null).
                    Vec3* emi_out = (out_vol_emission != nullptr
                                     && activeVolume->emission_scale > 0.0f)
                                    ? &vol_emission_local : nullptr;
                    sampled_scatter = sampleHeterogeneousScatter(
                        ray, t_medium_limit, *activeVolume, activeMedium,
                        channel, rng_state, t_vol, emi_out);
                } else {
                    const float xi_t = rng_next(rng_state);
                    t_vol = activeMedium.sampleFreePath(xi_t, channel);
                    sampled_scatter = (t_vol < t_medium_limit);
                }
                if (out_vol_emission != nullptr
                    && (vol_emission_local.x > 0 || vol_emission_local.y > 0 || vol_emission_local.z > 0)) {
                    *out_vol_emission = *out_vol_emission + throughput * vol_emission_local;
                }

                if (sampled_scatter) {
                    // ---- Medium scatter event ----
                    const Vec3 scatter_pos = ray.origin() + ray.direction() * t_vol;
                    Vec3 Tr;
                    Vec3 sigma_s_event = activeMedium.sigma_s;
                    Vec3 sigma_t_event = activeMedium.sigma_t;
                    if (heterogeneous) {
                        Tr = estimateTransmittance(ray, t_vol, activeVolume, activeMedium, rng_state);
                        const float density = activeVolume->sample_density(scatter_pos);
                        sigma_s_event = activeMedium.sigma_s * density;
                        sigma_t_event = activeMedium.sigma_t * density;
                    } else {
                        Tr = activeMedium.transmittance(t_vol);
                    }
                    const float Tr_avg      = fmaxf(spectrum_average(Tr), 1e-8f);
                    const float sigma_t_avg = fmaxf(spectrum_average(sigma_t_event), 1e-8f);
                    throughput = throughput
                               * (Tr * (1.0f / Tr_avg))
                               * (sigma_s_event * (1.0f / sigma_t_avg));

                    PathVertex& v    = path[n_vtx];
                    PathVertex& prev = path[n_vtx - 1];
                    v.type      = BDPT_VTX_MEDIUM;
                    v.delta     = false;
                    v.p         = scatter_pos;
                    v.n         = make_vec3(0,0,0);
                    v.ng        = make_vec3(0,0,0);
                    v.wo        = unit_vector(-ray.direction());
                    v.beta      = throughput;
                    v.Le        = make_vec3(0,0,0);
                    v.medium_id = activeVolumeIdx;
                    v.sigma_s   = sigma_s_event;          // local (density-scaled) for heterogeneous
                    v.sigma_t   = sigma_t_event;
                    v.phase_g   = activeMedium.g;
                    v.pdf_fwd   = convert_density_to_area(pdf_dir, prev, v)
                                * homogeneous_medium_dist_pdf(prev.p, v, volumeRegions, numVolumeRegions);
                    v.pdf_rev   = 0.0f;
                    ++n_vtx;
                    if (n_vtx > maxDepth) break;

                    // Sample HG phase direction. f == pdf for HG ⇒ no throughput multiplier.
                    const Vec3 wi_in = make_vec3(-ray.direction().x,
                                                 -ray.direction().y,
                                                 -ray.direction().z);
                    const float u1 = rng_next(rng_state);
                    const float u2 = rng_next(rng_state);
                    const Vec3  wi = activeMedium.samplePhaseHG(wi_in, u1, u2);
                    const float cos_t = dot(unit_vector(wi_in), unit_vector(wi));
                    const float pdf_phase = activeMedium.phaseHG(cos_t);
                    if (pdf_phase < 1e-10f) break;

                    if (n_vtx >= 4) {
                        float p_rr = fminf(luminance(throughput), 0.95f);
                        if (rng_next(rng_state) > p_rr) break;
                        throughput = throughput * (1.0f / fmaxf(p_rr, 1e-4f));
                    }
                    ray = Ray(scatter_pos, wi);
                    pdf_dir = pdf_phase;
                    continue;
                }

                // No medium scatter: attenuate by Tr to t_medium_limit, then exit volume or hit surface.
                Vec3 Tr;
                if (heterogeneous) {
                    Tr = estimateTransmittance(ray, t_medium_limit, activeVolume, activeMedium, rng_state);
                } else {
                    Tr = activeMedium.transmittance(t_medium_limit);
                }
                const float pdf = fmaxf(spectrum_average(Tr), 1e-8f);
                if (pdf < 1e-10f) break;
                throughput = throughput * (Tr * (1.0f / pdf));

                if (t_volume_exit + RT_EPS < t_surf) {
                    ray = Ray(ray.at(t_volume_exit + RT_EPS), ray.direction());
                    continue;
                }
                // Else fall through to surface processing.
            }
        }

        // ---------- Surface processing ----------
        if (!hit.hit) break;
        assignMaterialToHit(hit, numTriangles, triObjectIds,
                            objectMaterials, numObjectMaterials,
                            textures, numTextures);

        PathVertex& v    = path[n_vtx];
        PathVertex& prev = path[n_vtx - 1];
        const bool delta_here = is_delta_surface(hit);

        v.type      = BDPT_VTX_SURFACE;
        v.delta     = delta_here;
        v.p         = hit.p;
        v.n         = normalize(hit.normal);
        v.ng        = v.n;
        v.wo        = unit_vector(-ray.direction());
        v.hit       = hit;
        v.beta      = throughput;
        v.medium_id = findVolumeIndexAtPoint(v.p, volumeRegions, numVolumeRegions);
        v.Le        = hit.mat.emission;
        v.pdf_fwd   = convert_density_to_area(pdf_dir, prev, v);
        v.pdf_rev   = 0.0f;
        ++n_vtx;
        if (n_vtx > maxDepth) break;

        Vec3 wi;
        float pdf_sa;
        Vec3 ray_origin_offset = v.p;       // updated per-branch (dielectric needs side-aware offset)
        if (delta_here) {
            pdf_sa = 1.0f;
            if (is_dielectric(hit)) {
                // Snell + Schlick branching (dielectric glass).
                const Vec3  rayDir   = unit_vector(ray.direction());
                const bool  entering = hit.front_face;
                const float eta      = entering ? (1.0f / hit.mat.ior) : hit.mat.ior;
                const float cos_i    = fminf(-dot(rayDir, v.n), 1.0f);
                const float F        = schlick(cos_i, hit.mat.ior);
                const Vec3  refractD = refract_dir(rayDir, v.n, eta);
                const bool  tir      = (refractD.x == 0.0f && refractD.y == 0.0f && refractD.z == 0.0f);
                if (!tir && rng_next(rng_state) > F) {
                    wi = refractD;
                    ray_origin_offset = v.p - v.n * RT_EPS;       // step into transmitted side
                } else {
                    wi = reflect_dir(rayDir, v.n);
                    ray_origin_offset = v.p + v.n * RT_EPS;
                }
                throughput = throughput * hit.mat.specularColor;
            } else {
                // Perfect mirror.
                wi = reflect_dir(unit_vector(ray.direction()), v.n);
                ray_origin_offset = v.p + v.n * RT_EPS;
                throughput = throughput * (hit.mat.specularColor * hit.mat.kr);
            }
        } else {
            wi = SampleBRDF(hit, v.wo, rng_state, pdf_sa, /*allow_diffuse=*/true);
            if (pdf_sa <= 1e-10f) break;
            if (dot(wi, v.n) <= 0.0f) break;
            const Vec3  f     = EvaluateBRDF(hit, v.wo, wi);
            const float cos_i = fabsf(dot(v.n, wi));
            throughput = throughput * f * (cos_i / pdf_sa);
            ray_origin_offset = v.p + wi * RT_EPS;
        }

        if (n_vtx >= 4) {
            float p_rr = fminf(luminance(throughput), 0.95f);
            if (rng_next(rng_state) > p_rr) break;
            throughput = throughput * (1.0f / fmaxf(p_rr, 1e-4f));
        }
        ray = Ray(ray_origin_offset, wi);
        pdf_dir = pdf_sa;
    }

    return n_vtx;
}

// ---------------------------------------------------------------------------
// Camera subpath: build cam[0] = pinhole, then random_walk forward.
// ---------------------------------------------------------------------------
HYBRID_FUNC inline int generate_camera_subpath(
    const Ray& primary,
    int maxDepth,
    const int numTriangles,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    const int32_t* __restrict__ triObjectIds,
    const Material* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const TextureData* __restrict__ textures,
    int numTextures,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions,
    unsigned int& rng_state,
    PathVertex* path,
    Vec3* out_vol_emission = nullptr)
{
    if (maxDepth <= 0) return 0;

    PathVertex& cam = path[0];
    cam.type      = BDPT_VTX_CAMERA;
    cam.delta     = true;
    cam.p         = primary.origin();
    cam.n         = unit_vector(primary.direction());
    cam.ng        = cam.n;
    cam.wo        = make_vec3(0,0,0);
    cam.beta      = make_vec3(1.0f, 1.0f, 1.0f);
    cam.pdf_fwd   = 1.0f;
    cam.pdf_rev   = 0.0f;
    cam.medium_id = findVolumeIndexAtPoint(cam.p, volumeRegions, numVolumeRegions);
    cam.Le        = make_vec3(0,0,0);

    return random_walk(
        primary, /*throughput=*/make_vec3(1.0f, 1.0f, 1.0f), /*pdf_dir=*/1.0f,
        maxDepth, /*n_start=*/1,
        numTriangles, nodes, aabbs, triangles, triObjectIds,
        objectMaterials, numObjectMaterials,
        textures, numTextures,
        volumeRegions, numVolumeRegions,
        rng_state, path, out_vol_emission);
}

// ---------------------------------------------------------------------------
// Light subpath: emitter endpoint + first emission step + random_walk forward.
// ---------------------------------------------------------------------------
HYBRID_FUNC inline int generate_light_subpath(
    int maxDepth,
    const EmissiveTriInfo* __restrict__ emissiveTris,
    const float* __restrict__ emissiveCDF,
    int numEmissiveTris,
    float totalEmissiveArea,
    const Triangle* __restrict__ triangles,
    const int numTriangles,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const int32_t* __restrict__ triObjectIds,
    const Material* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const TextureData* __restrict__ textures,
    int numTextures,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions,
    unsigned int& rng_state,
    PathVertex* path)
{
    if (maxDepth <= 0 || numEmissiveTris <= 0 || totalEmissiveArea <= 0.0f) return 0;

    const float u_sel = rng_next(rng_state);
    const int   eidx  = binary_search_cdf(emissiveCDF, numEmissiveTris, u_sel);
    const EmissiveTriInfo& emi = emissiveTris[eidx];
    const Triangle& eTri = triangles[emi.triangleIdx];

    float u1 = rng_next(rng_state);
    float u2 = rng_next(rng_state);
    if (u1 + u2 > 1.0f) { u1 = 1.0f - u1; u2 = 1.0f - u2; }
    const Vec3 p_light = eTri.v0 * (1.0f - u1 - u2) + eTri.v1 * u1 + eTri.v2 * u2;
    const Vec3 n_light = normalize(emi.normal);

    const float pdfPos = 1.0f / totalEmissiveArea;
    const Vec3  Le_emit = emi.emission;

    PathVertex& y0 = path[0];
    y0.type      = BDPT_VTX_LIGHT;
    y0.delta     = false;
    y0.p         = p_light;
    y0.n         = n_light;
    y0.ng        = n_light;
    y0.wo        = make_vec3(0,0,0);
    y0.beta      = make_vec3(1.0f, 1.0f, 1.0f);
    y0.pdf_fwd   = pdfPos;
    y0.pdf_rev   = 0.0f;
    y0.Le        = Le_emit;
    y0.medium_id = findVolumeIndexAtPoint(p_light, volumeRegions, numVolumeRegions);

    const Vec3  wi_emit  = cosine_hemisphere_sample(n_light, rng_state);
    const float cos_emit = fmaxf(dot(n_light, wi_emit), 0.0f);
    const float pdfDir_emit = cos_emit * 0.31830988618379067f;
    if (pdfDir_emit <= 1e-10f) return 1;

    Vec3  throughput = Le_emit * (cos_emit / (pdfPos * pdfDir_emit));
    Ray   ray        = Ray(p_light + n_light * RT_EPS, wi_emit);

    return random_walk(
        ray, throughput, /*pdf_dir=*/pdfDir_emit,
        maxDepth, /*n_start=*/1,
        numTriangles, nodes, aabbs, triangles, triObjectIds,
        objectMaterials, numObjectMaterials,
        textures, numTextures,
        volumeRegions, numVolumeRegions,
        rng_state, path);
}

// ---------------------------------------------------------------------------
// Atomic float-Vec3 splat. On CUDA uses atomicAdd; on CPU uses regular add
// (single-threaded reference path doesn't need atomics).
// ---------------------------------------------------------------------------
HYBRID_FUNC inline void bdpt_atomic_splat(Vec3* dst, const Vec3& v) {
#ifdef __CUDA_ARCH__
    atomicAdd(&dst->x, v.x);
    atomicAdd(&dst->y, v.y);
    atomicAdd(&dst->z, v.z);
#else
    dst->x += v.x;
    dst->y += v.y;
    dst->z += v.z;
#endif
}

// ---------------------------------------------------------------------------
// Visibility test for a connection segment.
// ---------------------------------------------------------------------------
HYBRID_FUNC inline bool bdpt_visible(
    const Vec3& a, const Vec3& b,
    const int numTriangles,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles)
{
    Vec3 d = b - a;
    float dist = sqrtf(dot(d, d));
    if (dist < 1e-6f) return false;
    Vec3 dir = d * (1.0f / dist);
    Ray r(a + dir * RT_EPS, dir);
    HitRecord h{}; h.hit = false;
    SearchBVH(numTriangles, r, nodes, aabbs, triangles, h);
    return !h.hit || h.t >= dist - 2.0f * RT_EPS;
}

// ---------------------------------------------------------------------------
// MIS weight (Veach recursive ratio, PBRT-style)
// ---------------------------------------------------------------------------
//
// For strategy (s, t), temporarily install the four pdf overrides at
// {pt = cam[t-1], ptMinus = cam[t-2], qs = lig[s-1], qsMinus = lig[s-2]}
// reflecting how the connection edge would be sampled "from the other side",
// then walk both subpaths inward summing squared (rev/fwd) ratios.
HYBRID_FUNC inline float mis_weight(
    PathVertex* cam, int t,
    PathVertex* lig, int s,
    float totalEmissiveArea,
    const VolumeRegionGPU* __restrict__ volumeRegions = nullptr,
    int numVolumeRegions = 0)
{
    if (s + t == 2) return 1.0f; // only one strategy generates this length

    auto remap0 = [] (float f) { return f != 0.0f ? f : 1.0f; };

    // Bind shorthand pointers for the four "near-connection" vertices.
    PathVertex* qs      = (s >= 1) ? &lig[s - 1] : nullptr;
    PathVertex* pt      = (t >= 1) ? &cam[t - 1] : nullptr;
    PathVertex* qsMinus = (s >= 2) ? &lig[s - 2] : nullptr;
    PathVertex* ptMinus = (t >= 2) ? &cam[t - 2] : nullptr;

    // ---- Save originals so we can restore after weight evaluation ----
    float save_pt_pdfRev = 0.0f, save_pt_min_pdfRev = 0.0f;
    float save_qs_pdfRev = 0.0f, save_qs_min_pdfRev = 0.0f;
    bool  save_pt_delta  = false, save_qs_delta = false;

    // ---- pt.delta := false (connection vertex is treated as non-delta in MIS) ----
    if (pt) { save_pt_delta = pt->delta; pt->delta = false; }
    if (qs) { save_qs_delta = qs->delta; qs->delta = false; }

    // ---- pt.pdfRev: area pdf of pt sampled from the LIGHT side ----
    if (pt) {
        save_pt_pdfRev = pt->pdf_rev;
        if (s > 0) {
            // Light path extends qsMinus -> qs -> pt
            pt->pdf_rev = vertex_pdf(*qs, qsMinus, *pt, volumeRegions, numVolumeRegions);
        } else {
            // s = 0: pt itself IS the emitter (camera path hit emitter); rev pdf
            // is the area pdf of having picked this emitter point (1/totalArea).
            pt->pdf_rev = vertex_pdf_light_origin(totalEmissiveArea);
        }
    }
    if (ptMinus) {
        save_pt_min_pdfRev = ptMinus->pdf_rev;
        if (s > 0) {
            // Light path extends qs -> pt -> ptMinus
            ptMinus->pdf_rev = vertex_pdf(*pt, qs, *ptMinus, volumeRegions, numVolumeRegions);
        } else {
            // s = 0: emitter at pt emits in some direction; pdf for ptMinus
            // is the cosine-emission directional pdf converted to area.
            // Need a transient "light vertex view" at pt. Build minimal struct.
            PathVertex pt_as_light = *pt;
            pt_as_light.type = BDPT_VTX_LIGHT;
            ptMinus->pdf_rev = vertex_pdf(pt_as_light, /*prev=*/nullptr, *ptMinus, volumeRegions, numVolumeRegions);
        }
    }
    if (qs) {
        save_qs_pdfRev = qs->pdf_rev;
        // Camera path extends ptMinus -> pt -> qs
        qs->pdf_rev = vertex_pdf(*pt, ptMinus, *qs, volumeRegions, numVolumeRegions);
    }
    if (qsMinus) {
        save_qs_min_pdfRev = qsMinus->pdf_rev;
        // Camera path extends pt -> qs -> qsMinus
        qsMinus->pdf_rev = vertex_pdf(*qs, pt, *qsMinus, volumeRegions, numVolumeRegions);
    }

    // ---- MIS sum: walk camera subpath inward ----
    float sumRi = 0.0f;
    float ri = 1.0f;
    for (int i = t - 1; i > 0; --i) {
        ri *= remap0(cam[i].pdf_rev) / remap0(cam[i].pdf_fwd);
        if (!cam[i].delta && !cam[i - 1].delta) sumRi += ri * ri;
    }

    // ---- MIS sum: walk light subpath inward ----
    ri = 1.0f;
    for (int i = s - 1; i >= 0; --i) {
        ri *= remap0(lig[i].pdf_rev) / remap0(lig[i].pdf_fwd);
        bool delta_neighbor = (i > 0) ? lig[i - 1].delta : /*emitter delta?*/ false;
        if (!lig[i].delta && !delta_neighbor) sumRi += ri * ri;
    }

    // ---- Restore originals ----
    if (pt)      { pt->pdf_rev      = save_pt_pdfRev;     pt->delta = save_pt_delta; }
    if (ptMinus) { ptMinus->pdf_rev = save_pt_min_pdfRev; }
    if (qs)      { qs->pdf_rev      = save_qs_pdfRev;     qs->delta = save_qs_delta; }
    if (qsMinus) { qsMinus->pdf_rev = save_qs_min_pdfRev; }

    return 1.0f / (1.0f + sumRi);
}

// ---------------------------------------------------------------------------
// Connection: unweighted contribution of strategy (s, t).
// Returns 0 for skipped strategies (t<=1, etc).
// ---------------------------------------------------------------------------
HYBRID_FUNC inline Vec3 connect_paths(
    PathVertex* cam, int t,
    PathVertex* lig, int s,
    const int numTriangles,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions,
    unsigned int& rng_state)
{
    if (t < 2) return make_vec3(0,0,0);   // light tracing not supported (pinhole)

    // ---- s = 0: camera path naturally hits an emitter at cam[t-1] ----
    if (s == 0) {
        const PathVertex& v = cam[t - 1];
        if (v.type == BDPT_VTX_SURFACE && (v.Le.x > 0 || v.Le.y > 0 || v.Le.z > 0)) {
            return v.beta * v.Le;
        }
        return make_vec3(0,0,0);
    }

    PathVertex& pt = cam[t - 1];
    PathVertex& qs = lig[s - 1];

    // Connections at delta vertices have measure zero — skip.
    if (pt.delta || qs.delta) return make_vec3(0,0,0);

    // Visibility shadow ray.
    if (!bdpt_visible(pt.p, qs.p, numTriangles, nodes, aabbs, triangles))
        return make_vec3(0,0,0);

    // Direction from pt -> qs (and back).
    Vec3 d = qs.p - pt.p;
    float r2 = dot(d, d);
    if (r2 < 1e-20f) return make_vec3(0,0,0);
    float inv_r = rsqrtf(r2);
    Vec3 w_pt_to_qs = d * inv_r;
    Vec3 w_qs_to_pt = -w_pt_to_qs;

    // BSDF/phase at pt for (wo = back to cam-prev, wi = toward light side).
    Vec3 f_cam = vertex_eval_f(pt, w_pt_to_qs, pt.wo);
    if (f_cam.x <= 0 && f_cam.y <= 0 && f_cam.z <= 0) return make_vec3(0,0,0);

    float G = geometry_term(pt, qs);
    if (G <= 0.0f) return make_vec3(0,0,0);

    // Transmittance along the connection segment (homogeneous: closed-form;
    // heterogeneous: ratio tracking).
    Vec3 Tr = segment_transmittance(pt.p, qs.p, volumeRegions, numVolumeRegions, rng_state);

    // ---- s = 1: connect cam[t-1] to emitter endpoint y_0 ----
    if (s == 1) {
        float cos_l = dot(qs.n, w_qs_to_pt);
        if (cos_l <= 0.0f) return make_vec3(0,0,0);
        if (qs.pdf_fwd <= 0.0f) return make_vec3(0,0,0);
        return pt.beta * f_cam * G * Tr * qs.Le * (1.0f / qs.pdf_fwd);
    }

    // ---- s >= 2: full connection ----
    Vec3 f_lig = vertex_eval_f(qs, w_qs_to_pt, qs.wo);
    if (f_lig.x <= 0 && f_lig.y <= 0 && f_lig.z <= 0) return make_vec3(0,0,0);

    return pt.beta * f_cam * G * Tr * f_lig * qs.beta;
}

// ---------------------------------------------------------------------------
// Light tracing (t=1 strategies): project each light-path vertex onto the
// image plane and splat its contribution to that pixel via atomicAdd. MIS
// against the t>=2 strategies is handled by mis_weight() with t=1.
//
// Strategies splatted:
//   s = 1, t = 1:   emitter endpoint y_0 → camera. (Camera literally seeing
//                   the light surface — usually rare for narrow FOV but cheap
//                   to evaluate.)
//   s >= 2, t = 1:  light-bounce vertex y_{s-1} → camera. THIS is the win for
//                   caustics through delta surfaces and bright-light-through-
//                   dense-medium configurations: light walks hit a useful
//                   bounce point, then splat directly to where the camera
//                   sees them.
// ---------------------------------------------------------------------------
HYBRID_FUNC inline void light_tracing_splat(
    PathVertex* cam_path /* size >= 1; uses cam_path[0] only */,
    PathVertex* lig_path, int n_lig,
    int maxDepth,
    const Camera& camera,
    int img_w, int img_h,
    const int numTriangles,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions,
    float totalEmissiveArea,
    unsigned int& rng_state,
    Vec3* __restrict__ splat_buffer)
{
    if (n_lig <= 0 || splat_buffer == nullptr) return;

    const Vec3 cam_p = cam_path[0].p;
    const float A_z1 = camera.image_area_at_z1();
    if (A_z1 <= 0.0f) return;

    for (int s = 1; s <= n_lig; ++s) {
        // Total path length under (s, t=1) is s + 1 - 1 = s edges, s+1 vertices? Actually our
        // depth bound is s + t - 1 <= maxDepth (path edges). With t=1: s <= maxDepth.
        if (s > maxDepth) break;

        PathVertex& y = lig_path[s - 1];
        if (y.delta) continue;
        // s=1: y is the emitter endpoint. Connecting it directly to the camera renders
        // the emitter surface itself if it falls inside the FOV. Allowed.

        // Project onto image plane.
        float px, py, cos_e;
        if (!camera.world_to_pixel(y.p, px, py, cos_e)) continue;

        const int ix = (int)floorf(px);
        const int iy = (int)floorf(py);
        if (ix < 0 || ix >= img_w || iy < 0 || iy >= img_h) continue;

        // Direction from y back to the camera origin.
        Vec3 d = cam_p - y.p;
        float r2 = dot(d, d);
        if (r2 < 1e-12f) continue;
        const float inv_r = rsqrtf(r2);
        const Vec3  w_y_to_cam = d * inv_r;

        // Visibility through geometry.
        if (!bdpt_visible(y.p, cam_p, numTriangles, nodes, aabbs, triangles))
            continue;

        // BSDF/phase eval at y. For s=1 (y is emitter endpoint) we fall through
        // to the special-case formula below using y.Le directly.
        Vec3 contrib;
        if (s == 1) {
            // Emitter directly visible to camera. Diffuse cosine emission.
            const float cos_l = dot(y.n, w_y_to_cam);
            if (cos_l <= 0.0f) continue;
            // pdf_pos at y_0 is 1/totalEmissiveArea; folded into the Le contribution.
            if (y.pdf_fwd <= 0.0f) continue;
            // Contribution = Le × cos_l × Wi_pdf-equivalent
            contrib = y.Le * (cos_l / y.pdf_fwd);
        } else {
            const Vec3 f = vertex_eval_f(y, w_y_to_cam, y.wo);
            if (f.x <= 0 && f.y <= 0 && f.z <= 0) continue;
            const float cos_y = (y.type == BDPT_VTX_SURFACE)
                              ? fabsf(dot(y.n, w_y_to_cam)) : 1.0f;
            contrib = y.beta * f * cos_y;
        }

        // Pinhole importance: We·cos_e/pdf_camera = 1 / (A_z1 × cos³θ_e × r²).
        const float cos_e2 = cos_e * cos_e;
        const float cos_e3 = cos_e2 * cos_e;
        const float W = 1.0f / (A_z1 * cos_e3 * r2);
        contrib = contrib * W;

        // Transmittance along the y → camera segment.
        Vec3 Tr = segment_transmittance(y.p, cam_p, volumeRegions, numVolumeRegions, rng_state);
        contrib = contrib * Tr;

        // MIS against the t>=2 strategies producing a path of the same length.
        float w = mis_weight(cam_path, /*t=*/1, lig_path, s, totalEmissiveArea,
                             volumeRegions, numVolumeRegions);
        contrib = bdpt_sanitize(contrib * w);
        contrib = bdpt_clamp_firefly(contrib);
        if (!(contrib.x > 0 || contrib.y > 0 || contrib.z > 0)) continue;

        const int pix_id = iy * img_w + ix;
        bdpt_atomic_splat(&splat_buffer[pix_id], contrib);
    }
}

// ---------------------------------------------------------------------------
// BDPT integrator: full estimate for one primary ray.
// ---------------------------------------------------------------------------
HYBRID_FUNC inline Vec3 bdpt_li(
    const Ray& primary,
    int maxDepth,
    const Vec3& missColor,
    const int numTriangles,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    const int32_t* __restrict__ triObjectIds,
    const Material* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const EmissiveTriInfo* __restrict__ emissiveTris,
    const float* __restrict__ emissiveCDF,
    const int numEmissiveTris,
    const float totalEmissiveArea,
    unsigned int& rng_state,
    const TextureData* __restrict__ textures,
    int numTextures,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions,
    // t=1 light tracing splat buffer + camera projection. Pass nullptr to skip.
    Vec3* __restrict__ splat_buffer = nullptr,
    const Camera* camera = nullptr,
    int img_w = 0, int img_h = 0)
{
    // Subpath storage. Surface BDPT requires up to maxDepth+1 vertices per side.
    PathVertex cam_path[BDPT_MAX_DEPTH + 2];
    PathVertex lig_path[BDPT_MAX_DEPTH + 2];

    Vec3 vol_emission_cam = make_vec3(0, 0, 0);
    const int n_cam = generate_camera_subpath(
        primary, maxDepth + 1,
        numTriangles, nodes, aabbs, triangles, triObjectIds,
        objectMaterials, numObjectMaterials,
        textures, numTextures,
        volumeRegions, numVolumeRegions,
        rng_state, cam_path, &vol_emission_cam);

    const int n_lig = generate_light_subpath(
        maxDepth + 1,
        emissiveTris, emissiveCDF, numEmissiveTris, totalEmissiveArea,
        triangles, numTriangles, nodes, aabbs, triObjectIds,
        objectMaterials, numObjectMaterials,
        textures, numTextures,
        volumeRegions, numVolumeRegions,
        rng_state, lig_path);

    // t=1 splat: project each non-delta light vertex onto the image plane and
    // atomic-add its contribution to the splat buffer. Independent of n_cam
    // (a missed primary ray still has a useful light path).
    if (splat_buffer != nullptr && camera != nullptr && img_w > 0 && img_h > 0 && n_lig > 0) {
        light_tracing_splat(
            cam_path, lig_path, n_lig,
            maxDepth + 1, *camera, img_w, img_h,
            numTriangles, nodes, aabbs, triangles,
            volumeRegions, numVolumeRegions,
            totalEmissiveArea, rng_state, splat_buffer);
    }

    if (n_cam <= 1) {
        // Camera ray missed everything; show background.
        return missColor;
    }

    Vec3 L = bdpt_clamp_firefly(bdpt_sanitize(vol_emission_cam));

    // Iterate (s, t) with t >= 2 (skip light tracing) and s + t <= maxDepth + 1.
    for (int t = 2; t <= n_cam; ++t) {
        const int s_max = (maxDepth + 1) - t;
        const int s_lim = (s_max < n_lig) ? s_max : n_lig;
        for (int s = 0; s <= s_lim; ++s) {
            Vec3 C = connect_paths(cam_path, t, lig_path, s,
                                   numTriangles, nodes, aabbs, triangles,
                                   volumeRegions, numVolumeRegions, rng_state);
            if (C.x <= 0 && C.y <= 0 && C.z <= 0) continue;
            float w = mis_weight(cam_path, t, lig_path, s, totalEmissiveArea,
                                 volumeRegions, numVolumeRegions);
            // Sanitize first (kill any NaN/Inf produced by extreme MIS
            // pdf ratios in dense media), then bound luminance.
            Vec3 contrib = bdpt_sanitize(C * w);
            L = L + bdpt_clamp_firefly(contrib);
        }
    }

    return bdpt_sanitize(L);
}
