# Bidirectional Path Tracing (BDPT)

A Mitsuba/PBRT-style bidirectional path tracer added to MC-RayTracer alongside the existing `TraceRayIterative` unidirectional integrator. BDPT is **opt-in via `--integrator bdpt`** — the existing PT path is the default and is unchanged.

## What's supported

**Surfaces**
- Diffuse + Blinn-Phong glossy (via the existing `EvaluateBRDF` / `SampleBRDF` / `BRDFpdf` helpers)
- Perfect mirror (`kr > 0`, `kd = ks = 0`, `ior = 1`)
- Dielectric / glass (`ior > 1`, `kd = ks = 0`) — Schlick-Fresnel + Snell + total-internal-reflection

**Participating media (`volume_region` in scene JSON)**
- Homogeneous: inverse-CDF free-flight sampling, closed-form transmittance
- Heterogeneous density grids: delta tracking against the density majorant, ratio-tracked transmittance for connection segments
- Henyey-Greenstein phase function with arbitrary `g`
- Optional emissive media: blackbody emission from temperature/flame channels accumulated along the camera walk

**BDPT machinery**
- Camera + light subpaths share a single `random_walk` that handles surfaces and medium scatter together
- Strategies covered: `s = 0` (camera path hits emitter), `s = 1` (NEE-equivalent), `s ≥ 2` (full BDPT vertex connection)
- `t = 1` light tracing: each non-delta light vertex projects onto the image plane and atomic-adds its contribution to a splat buffer, merged into the output at the end of `render()`
- Veach recursive-ratio MIS with power(2) heuristic; analytic distance pdf folded in for homogeneous medium vertices
- Per-strategy `BDPT_FIREFLY_CLAMP` (default `50`) bounds each contribution's luminance — kills dense-media spikes; sanitize pass zeros any `NaN`/`Inf` before accumulation

Out of scope: light tracing through point lights, non-pinhole cameras, HDRI-as-environment-light for BDPT (the PT path's HDRI miss is preserved for PT runs).

## Build

```bash
cd MC-RayTracer
mkdir build && cd build
cmake -DENABLE_GPU=ON ..
make -j8
```

For a CPU-only build (slow but useful for debugging) drop `-DENABLE_GPU=ON`.

The executable is `render` in the build dir.

## Run

```bash
# Default: existing PT integrator
./render <scene.json> [-o output.png] [--nee-mode mis|brdf|area] [--denoise]

# Opt in to BDPT:
./render <scene.json> --integrator bdpt [-o output.png]
```

`--integrator pt` is the explicit default; `--integrator bdpt` switches the kernel to the BDPT path. All other flags work the same as before.

## Example scenes

Existing scenes under `assets/json_files/` cover the BDPT integrator's main regimes:

| Scene | What it exercises |
|---|---|
| `cornell_area_light.json` | Cornell box + area light + mirror sphere + emissive sphere — surface BDPT only |
| `cornell_volume_g0.json` | Glass spheres in homogeneous fog — dielectric + homogeneous medium |
| `cornell_smoke_taichi.json` | Cornell + heterogeneous smoke — delta tracking + ratio-tracked Tr |
| `cornell_smoke_embergen_120.json` | Dense embergen plume with hot emission — heterogeneous + volume emission |
| `cornell_smoke_embergen_47.json` | Plume with two emissive spheres at the base — t=1 light splatting earns its keep here |

```bash
# Surface-only Cornell box
./render ../../assets/json_files/cornell_area_light.json --integrator bdpt -o cornell_bdpt.png

# Homogeneous fog with glass
./render ../../assets/json_files/cornell_volume_g0.json --integrator bdpt -o fog_bdpt.png

# Dense smoke with hot emission
./render ../../assets/json_files/cornell_smoke_embergen_120.json --integrator bdpt -o smoke_bdpt.png
```

Compare-and-contrast against PT by re-running the same scene without `--integrator bdpt`.

## Compile-time knobs

Define before `#include "bdpt.h"` (or via `-D` on the compile line) to override defaults:

| Macro | Default | Effect |
|---|---|---|
| `BDPT_MAX_DEPTH` | `8` | Max number of vertices on each subpath. Path arrays are sized to `BDPT_MAX_DEPTH + 2`. The scene's `max_bounces` is clamped to this at the integrator entry. |
| `BDPT_FIREFLY_CLAMP` | `50.0f` | Per-`(s,t)` luminance clamp. `0` or negative disables (use for an unbiased reference). |

## Adding a volumetric scene

A `volume_region` entry in the scene JSON's `scene` array declares a participating medium:

```json
{
  "name": "fog",
  "type": "volume_region",
  "bounds_min": [-1, -1, -1],
  "bounds_max": [ 1,  1,  1],
  "medium": {
    "sigma_s": 0.6,
    "sigma_a": 0.01,
    "g": 0.0,

    "density_file":   "./assets/volumes/...raw",
    "density_resolution": [128, 128, 128],
    "density_format": "f32",
    "density_scale": 1.0,
    "majorant": 0.65,

    "emission_scale":      0.0,
    "emission_temperature": [400, 2200]
  }
}
```

`density_file` + the resolution/format triplet are optional — without them the region is homogeneous. `emission_scale > 0` activates blackbody emission from the temperature/flame channels (heterogeneous only).

## Notes on cost vs PT

BDPT is roughly 2–3× slower per sample than unidirectional PT — two random walks per sample plus `s × t` connection shadow rays. Variance is lower per sample, particularly in dense media and around delta materials, so equal-time it tends to be competitive or better. Disable the firefly clamp via `-DBDPT_FIREFLY_CLAMP=0` if you need a strict unbiased reference.
