#!/bin/sh
# Compile every Vulkan kernel (src/vulkan/shaders/*.comp) to SPIR-V for both wave sizes: wave32 (NVIDIA, AMD RDNA)
# into <out>/spv and wave64 (AMD GCN, --features vk-wave64) into <out>/spv64. Usage: tools/gen_spirv.sh [absolute
# out dir] (default src/vulkan; both sets are committed and tests/backend.rs checks they are up to date).
set -e
cd "$(dirname "$0")/../src/vulkan/shaders"
base=${1:-..}
gen() { # $1 = SUBGROUP_SIZE, $2 = F16, $3 = out dir
  mkdir -p "$3"
  for f in *.comp; do
    glslc -D"SUBGROUP_SIZE=$1" -D"F16=$2" --target-env=vulkan1.3 -O -o "$3/${f%.comp}.spv" "$f"
  done
}
gen 32 1 "$base/spv"
gen 64 0 "$base/spv64"
