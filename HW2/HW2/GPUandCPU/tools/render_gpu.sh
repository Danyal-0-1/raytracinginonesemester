#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_gpu"
OUT_DIR="${ROOT_DIR}/outputs"
BIN="${BUILD_DIR}/bvh_viz"
CSV="${OUT_DIR}/timing_gpu.csv"
BATCH_TXT="${OUT_DIR}/timing_gpu_batch.txt"

mkdir -p "${OUT_DIR}"
rm -f "${CSV}" "${BATCH_TXT}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DENABLE_GPU=ON
cmake --build "${BUILD_DIR}" -j

SPPS=(10 50 100)
PRESETS=(sunrise midday sunset night_moon night_nomoon)
REFERENCE_SCENE="${ROOT_DIR}/assets/json_files/reference_multispheres_midday.json"

batch_start=$(date +%s)

for preset in "${PRESETS[@]}"; do
  scene="${ROOT_DIR}/assets/json_files/sphere_env_${preset}.json"
  for spp in "${SPPS[@]}"; do
    out="${OUT_DIR}/gpu_env_${preset}_spp${spp}.png"
    echo "[GPU] scene=${preset} spp=${spp} -> ${out}"
    "${BIN}" "${scene}" --spp "${spp}" --output "${out}" --timing-csv "${CSV}"
  done
done

for spp in "${SPPS[@]}"; do
  scene="${ROOT_DIR}/assets/json_files/sphere_pointlight.json"
  out="${OUT_DIR}/gpu_pointlight_spp${spp}.png"
  echo "[GPU] scene=pointlight spp=${spp} -> ${out}"
  "${BIN}" "${scene}" --spp "${spp}" --output "${out}" --timing-csv "${CSV}"
done

for spp in "${SPPS[@]}"; do
  out="${OUT_DIR}/gpu_reference_midday_spp${spp}.png"
  echo "[GPU] scene=reference_midday spp=${spp} -> ${out}"
  "${BIN}" "${REFERENCE_SCENE}" --spp "${spp}" --env-enabled 1 --env-preset midday --output "${out}" --timing-csv "${CSV}"
done

batch_end=$(date +%s)
batch_seconds=$((batch_end - batch_start))

echo "backend,scene,preset,spp,width,height,bounces,seconds" > "${BATCH_TXT}"
echo "gpu,batch_total,batch_total,-1,0,0,0,${batch_seconds}" >> "${BATCH_TXT}"
echo "gpu,batch_total,batch_total,-1,0,0,0,${batch_seconds}" >> "${CSV}"
echo "GPU total batch render time: ${batch_seconds} seconds"
