# Environment Light MIS + JSON Extension

## MIS call flow (2-strategy: BSDF + Light)

- Light strategy: call `EnvironmentLight::Sample(...)` to get `wi` and `p_light`.
- Evaluate incoming radiance with `EnvironmentLight::Eval(wi)`.
- Query BSDF pdf for that same direction: `p_bsdf = bsdf.Pdf(wo, wi)`.
- Compute balance weight for light sample: `w_light = p_light / (p_light + p_bsdf)`.
- BSDF strategy: sample BSDF to get `wi` and `p_bsdf`; if miss ray reaches environment, call `EnvironmentLight::Eval(wi)`.
- For BSDF-sampled environment hit, call `EnvironmentLight::Pdf(wi)` to get `p_light`.
- Compute balance weight for BSDF sample: `w_bsdf = p_bsdf / (p_bsdf + p_light)`.
- Use solid-angle-consistent estimator per strategy: `L += Eval(wi) * f * cos_theta * w / pdf_strategy`.

## Top-level JSON extension (add alongside existing `environment` block)

```json
{
  "lights": [
    {
      "type": "environment_light",
      "enabled": true,
      "hdri_path": "./assets/hdris/old_main_sunrise.hdr",
      "intensity_scale": 1.0,
      "tint": [1.0, 1.0, 1.0],
      "importance_sampling_resolution": [1024, 512],
      "rotation_euler_deg": [0.0, 0.0, 0.0]
    }
  ]
}
```

