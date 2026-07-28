# Jetson AGX Orin (CUDA 12.6) workaround — read before using

This branch (`jetson-jp6-v0.26`) is **not a ready-to-build reference**. It is a
snapshot of the source changes I made to get `vllm serve nvidia/Cosmos3-Edge`
running on my own Jetson AGX Orin 64GB (JetPack 6.2.1, CUDA 12.6.68,
PyTorch 2.10.0). See the commit message on this branch for the full
technical explanation.

## Why cloning + building this branch will NOT work for you

The patches here rely on a **legacy `vllm/_C.abi3.so` binary that already
existed on my machine** from an older vLLM build (from before
[vllm-project/vllm#44334](https://github.com/vllm-project/vllm/pull/44334)
migrated CUDA kernels like `rotary_embedding` and `rms_norm` out of the `_C`
extension and into `_C_stable_libtorch`). That `.so` file:

- is a compiled binary, **not tracked in git** (it's gitignored, like all
  `*.so` files), so it was never pushed to this fork
- **cannot be rebuilt from the current source tree**: `csrc/torch_bindings.cpp`
  (the file that used to register those ops into `_C`) is now only ~39 lines
  and no longer registers `rotary_embedding`, `rms_norm`, etc. Those kernels
  now live under `csrc/libtorch_stable/`, which requires PyTorch >= 2.11 to
  compile (not available for JetPack 6 / CUDA 12.6 aarch64 as of this
  writing)

So if you clone this branch fresh and build it on your own Jetson AGX Orin,
you will hit the same failure I originally had (`AttributeError:
'_OpNamespace' '_C' object has no attribute 'rotary_embedding'`) — my `_C`
extension only "works" here because of a leftover artifact from an earlier
build of this repo, not because of anything in this branch's source diff.

## What this branch is actually useful for

- Evidence that Cosmos3-Edge **can** run well on AGX Orin (benchmarked
  ~165 tok/s @ concurrency=4, ~45 tok/s @ concurrency=1, TTFT ~60ms) once the
  PyTorch 2.11 requirement is met
- A reference diff of what changes were needed on top of vLLM `main`
  (`568afb3a1`) once a working `_C_stable_libtorch` build is available
- Context for the upstream discussion in
  [NVIDIA-AI-IOT/jetson-ai-lab](https://github.com/NVIDIA-AI-IOT/jetson-ai-lab)
  about PyTorch 2.11+ availability for JetPack 6 / CUDA 12.6

## What would actually make this reproducible

Rebasing these Jetson-specific changes onto the last vLLM commit **before**
#44334 (where `csrc/torch_bindings.cpp` still registered the full kernel set
for CUDA) would let anyone build a working `_C` extension from source on
CUDA 12.6 / PyTorch 2.10, without needing my leftover binary. That hasn't
been done here yet.

Debugged and patched with AI assistance (Claude Code).
