# AMD GPUs

All benchmarks in the README were run on an NVIDIA RTX 3060. AMD GPUs run the Vulkan backend
(`--backend vulkan`), and everything known about them comes from user reports (RX 9070 XT on Linux with RADV, and on
Windows with AMD's driver). Expect problems; this page lists the known ones and what to send when you hit a new one.

Needs a Vulkan 1.3 driver. By generation:

| GPU | Build |
| --- | --- |
| RDNA 1 and newer (RX 5000, 6000, 7000, 9000) | default: `cargo build --release` |
| GCN 4 and 5 (RX 400/500, Polaris, Vega) | `cargo build --release --features vk-wave64` (wave64 subgroups, no f16) |
| older GCN (Tahiti to Fury) | untested; the wave64 build is the one to try |

The wave64 build runs every kernel as two 32-lane warps per 64-lane wave and does the f16 KV-cache conversions with
integer ops; it is otherwise the same binary. GCN has no cooperative matrices (prefill uses the slower `dp` kernels,
as on RDNA) and no device-coherent memory (the expert mailbox is cached by the GPU until a submission ends, so decode
runs pipelined; see below). GCN also has no int8 dot-product instruction: prefill GEMMs therefore dequantize to f16
(the fast choice; `STRATA_MMQ=<types>` forces the exact-but-slower int8 path), while decode GEMVs use driver-emulated
int8 dots. The fp64 RoPE runs at 1/16 rate on GCN — correct, mildly slow. One measured data point
(RX 580 8 GB, Qwen3.6-35B-A3B Q3_K_M, 3816/10240 experts resident): 26 tok/s
decode with experts on the CPU, 56 tok/s prefill at `--pp 128` (short prompts pay the expert staging cost; at
`--pp 2048` the same model prefills at 278 tok/s).

## First line of the output

Every run on Vulkan starts with a line like:

```
vulkan: AMD Radeon RX 9070 XT (radv Mesa 26.2.5), device-coherent memory yes, cooperative matrix f16 yes int8 no, conditional rendering yes
```

- `device-coherent memory` must be `yes` for models whose experts do not all fit in VRAM (see below). GCN never has it:
  there the GPU caches the mailbox until a submission ends, so a wave64 build pipelines decode instead (one layer per
  submission, which costs a little); `STRATA_VK_NO_PIPE=1` turns that off.
- `cooperative matrix` speeds up prefill. `int8 no` is normal on RDNA.
- `conditional rendering` lets decode recover when the CPU expert worker is slow. With `no`, a slow worker is an error.

## Known issues

**Slow decode with experts on the CPU, and "GPU waits for the CPU expert worker that timed out and resumed: N".**
While the CPU computes the experts that are not in VRAM, the GPU polls a small mailbox buffer for the answer. On AMD the
GPU caches that buffer unless it comes from device-coherent memory (`VK_AMD_device_coherent_memory`), and then it misses
the answer and waits until it gives up. RADV hides the extension on RDNA4 (RX 9000) unless the driconf option
`radv_device_coherent_memory` is set. Strata sets it before it opens Vulkan (unless you set it yourself) and prints a
warning when an AMD GPU still has no device-coherent memory. If the first line says `device-coherent memory no`, send
the output of `vulkaninfo | grep -i coherent`. A few resumed waits on a busy CPU are normal; hundreds are not. On GCN
(RX 400/500, Vega) the answer never becomes visible at all during a submission, so a wave64 build pipelines decode
(one layer per submission) and these resumes should not appear; hundreds still mean a bug.

**The GPU resets during a long prefill.** amdgpu resets a submission that runs for more than 10 s. Prefill submits
each layer separately to stay under that. If it still happens, try a smaller `--chunk`.

**"the forward pass produced NaN or infinite logits".** Seen once on Qwen3-Next 80B with a 15K prompt. Rerun with
`STRATA_NAN_TRACE=1`, which prints per-layer magnitudes and the first layer with a non-finite value, and attach the
output.

**A model split over two cards (`--device 0,1`) is slower than expected.** Each card waits while the other works, and
amdgpu clocks it down in those gaps. `echo high > /sys/class/drm/cardN/device/power_dpm_force_performance_level` (as
root, for each card; `auto` restores the default) keeps the clocks up: 44 to 57 tok/s decode in one two-card report.
The per-device times that `generate` prints on a split show whether a card is slow.

**Load fails for lack of VRAM.** The desktop on the same card takes memory, and the driver's free-memory estimate can
be more than it will allocate. Lower `--ctx`, use `--kv q8`, or a smaller `--chunk`.

## Switches for testing

Environment variables; set any of them to turn the feature off:

- `STRATA_VK_NO_COOP`: no cooperative matrices.
- `STRATA_VK_NO_COND`: no conditional rendering.
- `STRATA_VK_NO_TQ`: no separate transfer queue for expert uploads.
- `radv_device_coherent_memory=false`: undo the RADV option strata sets.

If one of these makes a problem go away, say which one in the report.

## What to send in an issue

- GPU model, OS, driver and version (Mesa version on Linux).
- The output of `vulkaninfo --summary`.
- The full output of the failing command, including the first `vulkan:` line.
- If decode is slow: `strata bench --model <file> --gpu --backend vulkan --pp 1024 --tg 128` with the same model.
