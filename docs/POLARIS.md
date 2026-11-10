# POLARIS.md — GCN4 (RX 580) findings

Some AI agent notes (Mesa 26.2.5-arch3.1 radv POLARIS10, wave64 build).

## Overview

Porting the Vulkan backend to GCN4 (RX 580 / POLARIS10, gfx8, wave64 build, Mesa 26.2.5 radv) and tuning
it against llama.cpp on a 35B-A3B MoE (IQ3_S: 40 MoE layers + an MTP head, 256 experts, top-k 8, ~1.18 MB
per expert). The box: 8 GB VRAM (~35% of experts resident), Ryzen 2700X + DDR4-3200, and three missing
features that define the whole effort - **no `VK_AMD_device_coherent_memory`, no cooperative matrices, no
conditional rendering** - plus a GTT path the GPU can only read at 2.6 GB/s vs 181 GB/s from VRAM.

Three walls, in the order we hit them:
- **Prefill hangs** the ring past ~8k context (a Mesa gfx8 / hardware fault below our kernel); worked
  around by capping the prefill chunk to 2048.
- **Prefill is staging-bound**, not kernel-bound: the GTT->VRAM expert copies dominate the wall, so missed
  experts run on the CPU instead of being staged (cap ~24).
- **Decode is GPU-job-boundary-bound**: the non-coherent mailbox forces one submission (a GPU *job*) per
  MoE layer, so 40 host->GPU round-trips at ~1.65 ms each are the floor. Batching (MTP) does not amortize
  them, and the dynamic per-expert cache is what *creates* them.

The through-line: **the dynamic expert cache is a coherent-GPU design.** On non-coherent GCN4 the per-layer
job boundary is the decode floor and the right tool is a static whole-layer split (what llama.cpp does);
the cache pays off only where boundaries are free (RDNA4 + `radv_device_coherent_memory`, where the code
already runs unpipelined). The sections below are the detailed record; `## Open points` collects what is
left. Diagnostic probes are env-gated and committed: `STRATA_STEP_PROF`, `STRATA_WAIT_PROF`,
`STRATA_MTP_PROF`, `STRATA_PIPE_PROF`, `STRATA_VK_PROF`.

## The long-prefill attention hang (solved by workaround, Nov 2026)

**Symptom.** With `--chunk 4096`, the first prefill attention at position >= ~8192 (third chunk of a
12288-token prompt) either wedges the ring (kernel kills the job after 10 s: dmesg "ring gfx timeout,
but soft recovered"; radv dump: CP parked at the `attn_prefill_dp` dispatch, no VM faults) or, in
plain mode, completes and poisons the run (NaN logits at the first decode step). Probabilistic in
plain mode (pp 20480 "mostly" fails, pp 12288 passes), 100% under `RADV_DEBUG=hang`. Chunk 512-2048:
never, all day, both modes. Chunk 2560/3072: sometimes, later. llama.cpp at `-b 4096 -ub 4096`
(`-fa 1`) on the same model/context/GPU: never.

**Ruled out** (each with a run): transfer-queue overlap (`STRATA_VK_NO_TQ=1` hangs too); semaphore
protocol (audited); VRAM pressure (EXPERT_MB variants); launch profile (512/16 KB, 256/8 KB LDS, and
older kernels all hang); slowness (attn averages 16 ms, `STRATA_VK_PROF`); DPM/clocks (full clock,
games and llama.cpp fine); 4 GB VA boundary (no BO straddles anything, max VA 11.8 GB); corrupted
step buffer (`STRATA_STEP_DEBUG` probes clean); garbage arguments (kernel-side record in mailbox
memory matches; radv dump shows the wedged dispatch's user data/spill slot carry the correct
pointers, incl. a +32 MB split offset); compiler backend (ACO only - LLVM is disabled in release
Mesa); kernel logic (loops bounded, barriers uniform, clamped to cap).

**Decisive experiment.** Splitting the 4096-query attention into two correct 2048-query sub-launches
(`q0` push constant, stepped q/qg/out/qm) still hangs at the same place - the dumps prove the
sub-launches arrive at the CP with correct arguments, and the identical shape inside a chunk-2048
run never hangs. The trigger is the chunk context (slab sizes, dispatch stream, VA layout of a
4096-chunk run), not the attention launch's shape.

**Workaround.** `Backend::prefill_chunk_cap()` - the Vulkan dp path (no cooperative matrices, i.e.
GCN) reports 2048; the host defaults `--chunk` to min(calibrated, cap). Explicit `--chunk 4096`
still runs, with a warning. llama.cpp's default n_batch is 2048 too. Cost (RX 580, IQ3_S 35B-A3B):
pp 12288 213 -> 190 tok/s (-11%), pp 20480 152.7, decode 9.8 -> 10.9 (smaller slabs, +200 resident
experts). Rock solid across many runs at pp 12288/20480, plain and `RADV_DEBUG=hang`.

**Safety nets kept in the kernel** (`attn_prefill_dp.comp`): the key loop clamps to `step.v[1]`
(KV capacity; 0 = off), and one workgroup records the received arguments plus started/finished flags
into a mailbox-memory debug buffer before starting. Host side: `attn_args_dbg` compares the record
against what was sent at the next router readback; `STRATA_TRACE=1` adds a dispatch log and a
watchdog that classifies a hung attention (never started / started never finished) before the
kernel's 10 s kill; `STRATA_ATTN_SPLIT` overrides the query-split cap (2048 default; a no-op at the
capped chunk size).

**Still open.** Root cause below our kernel (Mesa gfx8 path or hardware) - we are the upstream for
the wave64 port, and the evidence says correct work delivered correctly still wedges. Repro recipe
above; `RADV_DEBUG=hang` makes it deterministic. A raw devcoredump (sudo run; umr segfaults) with
the wave PC would be the next data point if it ever bites again.

## Prefill is staging-bound; the CPU expert tail (Nov 2026)

`STRATA_VK_PROF` + `STRATA_PF_PROF` at chunk 2048, IQ3_S 35B-A3B: kernels are ~2 s of a 65 s prompt (gemm_f_dp 54%,
attn_prefill_dp 41% OF THE KERNELS) - the wall is "GPU wait": the unprofiled GTT->VRAM staging copies, 2.6-4.1 GB per
chunk at ~0.4 GB/s effective. Resident experts were 5160/10496 (49% of slots, 97% of token selections); the 3% miss
tail spreads over ~35 experts per layer, each a full 1.8 MB copy.

Fix: missed experts with <= `Backend::prefill_cpu_moe_cap()` (24 on the dp path) token slots run on the CPU from the
mapped weights (`cpu_moe::expert_rows`, flat parallel pass, results uploaded straight into the `ye` slot rows, which
are contiguous per expert); fatter missed experts keep staging. Staging costs a copy per expert, the CPU pass an FFN
per token - the crossover on this box (Ryzen 2700X, DDR4-3200) is near 24-32 tokens; above 48 the serialized CPU pass
loses to staging (pp 12288, vram-margin 2800 = 128k-KV emulation, 83% hit: stage 174.6, cap 24 193.0, cap 64 151.6;
at 97% hit: stage 189.9, cap 64 213.2). Decode gained ~0.4 tok/s too (pre-placement no longer competes with staging).

Open: calibrate the cap per machine (end-to-end sweep in `calibrate`, since isolated rates misjudge it - staging
overlaps the GEMMs and small interleaved copies run slower than a lone 16 MB one: the isolated formula said 6, the
sweep and the manual sweep both say 24-32); thread count for the CPU pass (16 SMT slightly over 8 physical at the
sweet spot; CCX pinning not worth it). GTT-direct is DEAD: `STRATA_GTT_PROBE` says kernels read host-visible memory
at 2.6 GB/s vs 181 GB/s from VRAM (70x) - the GPU reaching into RAM is not a path on this chip; staging + the CPU
tail is the design. Decode is neither worker- (14.3% at 67% hit) nor residency-bound (11.1 tok/s at 97%
hit vs 10.6 at 67%): the ~90 ms/token is the per-layer submit + worker-wait chain of `expert_pipeline`,
now explained in "Decode is GPU-job-boundary-bound" below.

## Decode is GPU-job-boundary-bound (non-coherent GCN4) (Nov 2026)

**Confirmed.** The "Open" item above is the whole story: decode's ~84 ms/token is 40 MoE layers x ~1.65 ms
of per-layer `wait_done` = ~66 ms of GPU *job* round-trips - not memory, not the worker, not the readback.
Host wall-clock (`STRATA_STEP_PROF`): verify_run ~69 ms + argmax-get ~14 ms. Per-layer split
(`STRATA_WAIT_PROF`): submit->REQ ~1400 us, worker turn ~200 us, host wake ~57 us. The worker hot-spun
~940k times during submit->REQ (never descheduled) and computes ~0.2 ms/layer; every GPU dispatch is sub-10
us (`STRATA_VK_PROF`). So the ~1.4 ms is the cost of *ending* a submission (the coherency point) and
*starting* the next amdgpu job, paid once per MoE layer.

**Why a boundary per layer even on a hit.** `moe_route.comp` raises the request flag *unconditionally*
(word 0, line 35) and the host `wait_done`s every MoE layer (`forward_pipelined`, gpu_model.rs:768); the
GPU combine already skips waiting on a hit (`moe_down_combine:53` `ok = !any || mb_wait`), so the 1.65 ms
is purely the host-side job split. GCN caches the GTT mailbox lines until a submission ends (mod.rs
`expert_pipeline` comment), so each host->GPU mailbox write needs its own job boundary to become visible -
hence one submission per MoE layer. (The "41 layers" above = 40 MoE + the MTP head.)

**The law.** On non-coherent memory, *boundaries ~= #(layers where the GPU needs a non-resident expert)*.
The dynamic per-expert cache maximizes this (partial residency is its whole point), so it is the wrong tool
here: at 35% residency (3568/10240) it pays ~40 boundaries. It is a *coherent-GPU* design - on RDNA4 +
`radv_device_coherent_memory` the code already runs unpipelined (one submission, in-kernel `mb_wait`), no
per-layer round-trips.

**Batching does not amortize it (MTP, RUN 18).** `--mtp 3` verifies 4 tokens in one forward, but
`verify(nt=4)` = 238 ms vs `verify(nt=1)` = 69 ms - the per-layer `wait_done` grows ~linearly with nt
(worker-wait 66 -> 225 ms), so the 40 round-trips are fixed and the per-layer cost scales with tokens.
Acceptance was high (92.5%, 186/201) and the draft phase cheap (53 ms vs verify 274 ms), yet decode stayed
11.4 vs 11.9 tok/s. Prefill's ~6x batching is a *different algorithm* (expert-major grouped GEMM, weights
read once, no per-layer mailbox, all-resident) that the decode path does not use and cannot use at partial
residency. Ruled out for decode (each with a run): memory latency, GPU idle vs nvtop, uncached mailbox,
keep-alive spin, BO count, clock/power, worker round-trip / starvation / re-read, transport mechanism
(mailbox 94 us ~ DMA 85 us), wake-from-idle, and the predictive fast path (a wrong guess costs the 1 s
combine give-up + a partial re-run).

**Fewer boundaries: what works and what collapses.** GPUs are Turing-complete (route top-k, `mb_wait`'s
500k spin loop and give-up branch, the combine's residency branch all decide/branch today), so
"hit->continue / miss->stop" per layer is not the blocker; only conditional rendering is missing on Polaris
(emulatable by early-exit dispatches). The collapse ideas:
- *Zone A - pin whole layers (works, unimplemented).* Pin the first N MoE layers fully in VRAM and skip
  `submit_nowait`/`wait_done` for them (no mailbox write => no coherency point => fold into one submission).
  Saves ~1.65 ms per pinned layer; ~14 layers at 35% residency => ~16 tok/s. The only implementable win on
  this card.
- *Zone B - hybrid cached layers with fall-off-to-CPU (collapses).* A miss at layer X forces X+1..end onto
  the CPU (data dependency), and a layer fully hits only if all top-k=8 experts are resident (~0.78^8 ~=
  14%, ~30% with correlation), so a token survives ~1-2 layers before one miss dumps it to CPU; B then
  degenerates to a static split at ~A+1, i.e. llama.cpp. Layer-bundling fails the same way (mid-bundle
  coherency + the 1 s combine give-up).

**Bottom line.** On non-coherent GCN4 the per-layer GPU job boundary is the decode floor; the dynamic
expert cache is designed for coherent GPUs. Use static whole-layer pinning (llama.cpp-style) or a coherent
GPU here.

## Open points

- **Prefill hang root cause** is below our kernel (Mesa gfx8 path or hardware): correct work delivered
  correctly still wedges past ~8k. We are the upstream for the wave64 port. Next data point would be a raw
  devcoredump (sudo; `umr` segfaults) with the wave PC. `RADV_DEBUG=hang` makes it deterministic.
- **CPU-MoE cap is not per-machine.** The prefill tail cap (24) is a manual-sweep value; isolated rates
  misjudge it (they said 6, the sweep says 24-32). Should be an end-to-end sweep in `calibrate`.
- **CPU expert-pass threading** (16 SMT marginally over 8 physical at the sweet spot; CCX pinning not worth
  it) is unconfirmed across machines.
- **Zone-A pinning is unimplemented** - the one implementable decode win on this card (pin the first N MoE
  layers, skip their mailbox): ~1.65 ms saved per pinned layer, ~16 tok/s at N~14. Needs a route variant /
  host path that skips `submit_nowait`+`wait_done` for fully-resident layers.
- **Should non-coherent GPUs default to static whole-layer pinning** instead of the dynamic cache? The
  analysis says yes on this card; not yet a code decision.
- **The ~1.4 ms submit->REQ micro-cause** (amdgpu job scheduling vs GTT invalidation at the boundary) is
  not isolated further; the existing probes cannot separate the two.
- **argmax-get ~14 ms/token** (logits readback + lm_head) is ~17% of decode and is not overlapped with the
  next step; a possible secondary target once the boundary floor is addressed.

## Misc

- `RADV_DEBUG=hang` re-uploads each shader (self-overlapping `DMA_DATA`, write-confirm disabled)
  before every dispatch and serializes all dispatches: useful for exact dispatch attribution, but it
  swallows our watchdog (the submit waits per dispatch) and its timing *stabilizes* the bug rather
  than causing it.
- `umr` in the radv dump segfaults (even as root) on this kernel; `dmesg.log` needs a sudo run.
- radv BO dumps need `RADV_DEBUG=allbos` for `bo_ranges.log`; `bo_history.log` works without it.
