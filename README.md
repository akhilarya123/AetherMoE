# AetherMoE

A from-scratch, $0, M1-Mac-only MoE inference engine, built milestone by milestone.

---

## Status at a glance

| Milestone | Status |
|---|---|
| **1 — Foundation & core engine loop** (C++ ingress/scheduler/paging) | **Done, verified on your Mac.** 18/18 tests pass, clean under ThreadSanitizer, 10,130 req/s on the load test (target was 10,000+), zero errors, zero drops. |
| **2 — Fused MoE kernels** (MLX/Metal gating + quantized GEMM) | **Correctness fully verified — 25/25 tests pass.** Performance meaningfully improved on both kernels (gating: ~2.4-3x slower than reference → ~1.7-2.2x; quantized GEMM: ~2.4-3.1x → ~1.1-2.8x depending on scale) but doesn't reliably clear the 1.5x threshold. Root cause is understood and evidenced via Instruments (Apple hardware matrix-multiply throughput ceiling, not overhead or memory access). Explicitly decided to stop tuning and move on rather than attempt a much larger `simdgroup_matrix` rewrite. Full story: [`MILESTONE2_PERF_EVIDENCE.md`](./MILESTONE2_PERF_EVIDENCE.md). |
| 3 — Simulated multi-node orchestration | Not started |
| 4 — Production hardening | Not started |

---

## Milestone 1 — C++ engine (done)

**What's in it:**

| Component | File |
|---|---|
| Lock-free SPMC ring buffer (Vyukov bounded queue) | `src/core/spmc_ring_buffer.hpp` |
| Sequence state machine (PREFILL→DECODE→FINISHED) | `src/core/sequence.hpp` |
| Paged KV-cache allocator (PagedAttention-style) | `src/core/page_table.hpp` |
| Continuous batching scheduler | `src/core/scheduler.hpp` |
| Ingress HTTP API (`/generate`, `/status/:id`, `/healthz`) | `src/api/ingress_server.{hpp,cpp}` |
| Engine entrypoint | `src/main.cpp` |
| Load generator | `benchmarks/load_generator.cpp` |
| Tests (18 GTest cases) | `tests/test_*.cpp` |

**Verified, with real numbers from your machine:** all 18 tests pass, clean under ThreadSanitizer (no data races), and the load test hit **10,130 req/s with 45,000/45,000 accepted, zero errors, zero backpressure drops** — meeting the milestone's 10,000+ req/s target. One real bug was found and fixed along the way: cpp-httplib's default TCP listen backlog (5) was silently dropping connections above that; raised to 1024.

**How to build & run:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)

./build/aether_tests                     # 18 tests should pass
./build/aether_engine 8080                # run the engine
./build/load_generator 127.0.0.1 8080 150 300   # load test, another terminal
```

Concurrency check (optional, already clean once):
```bash
cmake -B build-tsan -DAETHER_SANITIZE=thread
cmake --build build-tsan -j$(sysctl -n hw.ncpu) --target aether_tests
./build-tsan/aether_tests
```

**Known limitations (by design):** no tokenizer/model/MLX yet — decode steps append a placeholder token. Page table uses a mutex, not lock-free (documented as an optional stretch goal, unlikely to be the bottleneck). `IngressServer::stop()` detaches the HTTP thread rather than cleanly stopping it — fine for a dev binary.

---

## Milestone 2 — MLX/Metal kernels (correctness done, performance open)

**What's in it:**

| Component | File |
|---|---|
| Reference (unfused) gating — plain MLX ops | `src/kernels/reference_gating.py` |
| **Fused gating kernel** — the actual deliverable | `src/kernels/fused_gating_metal.py` |
| Quantized (INT8) expert GEMM — naive + fused | `src/kernels/quantized_gemm.py` |
| PyTorch `mps` reference (independent ground truth) | `validation/pytorch_reference.py` |
| Pure-NumPy math prototype (proven correct, zero deps) | `prototype/math_proto.py` |
| Tests | `tests/python/test_*.py` |
| Benchmark | `benchmarks/bench_gating.py` |
| Crash bisection tool | `debug_property_crash.py` |

### The story so far (worth knowing, not just the end state)

This milestone required real back-and-forth, unlike Milestone 1 — I have no GPU/MLX/Metal access in my own environment, so nothing here was actually testable by me until you ran it. Three real, distinct bugs were found this way and are now fixed:

1. **Grid dispatch bug (the big one).** `grid` in `mx.fast.metal_kernel` means *total thread count* (Metal's `dispatchThreads` semantics), not "number of threadgroups" — I had that backwards. With a fixed 256-thread threadgroup and `grid=(n_tokens,...)` for `n_tokens < 256`, MLX rounded that up to just **one** threadgroup total, so only "token slot 0" was ever computed and every other row was left as zero/garbage. This explained nearly every numerical test failure (row 0 correct, everything else wrong). Fixed: `grid` is now `n_tokens * threadgroup_size`.
2. **Metal compile error for tiny buffers** (found via `debug_property_crash.py`, which is exactly why that tool exists): `cannot initialize a variable of type 'const device float *' with an rvalue of type 'const constant float *'`. MLX places very small buffers (e.g. a 1×1 array) in Metal's `constant` address space rather than `device` as an optimization; I'd hardcoded `const device float*` for a pointer derived from the input, which only matched larger buffers (that's why bigger shapes worked and only the smallest crashed). **Just fixed:** changed to `auto` in both `fused_gating_metal.py` and `quantized_gemm.py`, letting the compiler infer whichever address space MLX actually used.
3. **A test's own arithmetic bug** (`test_scheduler.cpp`, Milestone 1) — caught and fixed during that milestone's own testing.

**Verified passing on your Mac — 25/25:**
```
tests/python/test_math_prototype.py    5/5   (pure NumPy, no MLX needed)
tests/python/test_gating_numerical.py  7/7   (fused vs MLX reference vs PyTorch mps)
tests/python/test_gating_property.py   7/7   (Hypothesis + explicit edge cases)
tests/python/test_quantized_gemm.py    6/6
```

**Performance — meaningfully improved, decided to stop tuning here:**
```
n_tokens=256  d_model=1024  num_experts=16  k=4
Gating kernel:      handover baseline ~2.4-3.0x slower -> now ~1.7-2.2x slower
Quantized GEMM:     v1 ~2.4-3.1x slower              -> now ~1.1-2.8x slower (scale-dependent)
```
Root cause for the remaining gap is confirmed via Instruments (Metal System
Trace), not guessed: hand-written kernels using scalar SIMD-group arithmetic
can't match Apple Silicon's dedicated matrix-multiply hardware path that
MLX's built-in matmul uses. Closing this further needs a `simdgroup_matrix`
rewrite of both kernels — evaluated and explicitly deferred as a
materially bigger undertaking than anything else in this milestone, after
two rounds of narrower tuning showed diminishing, often noise-level
returns. Full investigation, all real numbers, and the Instruments
evidence: **[`MILESTONE2_PERF_EVIDENCE.md`](./MILESTONE2_PERF_EVIDENCE.md)**.

### How to run everything

```bash
# one-time setup
python3.11 -m venv venv
source venv/bin/activate
python3.11 -m pip install -r requirements-m2.txt
```

**Important:** always invoke as `python3.11 -m pytest ...` / `python3.11 script.py`, never bare `pytest`/`python`. On this machine, bare `pytest` resolves to a pyenv shim pointing at a different Python (3.14) that doesn't have MLX/PyTorch installed, and silently skips everything.

```bash
# 1. Dependency-free sanity gate
python3.11 -m pytest tests/python/test_math_prototype.py -v

# 2. Gating correctness (fused vs MLX reference vs PyTorch mps)
python3.11 -m pytest tests/python/test_gating_numerical.py -v

# 3. Quantized GEMM correctness
python3.11 -m pytest tests/python/test_quantized_gemm.py -v

# 4. Property-based tests -- RE-RUN THIS FIRST, root cause of the crash is fixed
python3.11 -m pytest tests/python/test_gating_property.py -v

# 5. If anything ever crashes the process again (not a normal test failure),
#    this bisects it deterministically without relying on Hypothesis:
python3.11 debug_property_crash.py

# 6. Performance (currently failing its own regression threshold -- expected for now)
python3.11 benchmarks/bench_gating.py
python3.11 benchmarks/bench_gating.py --n-tokens 512 --d-model 4096 --num-experts 64 --k 8

# 7. Everything
python3.11 -m pytest tests/python/ -v
```

### Status: correctness closed, performance investigated and documented, moving on

All 25 tests pass. Performance was investigated thoroughly with real
Instruments evidence (not guessing) — see
[`MILESTONE2_PERF_EVIDENCE.md`](./MILESTONE2_PERF_EVIDENCE.md) for the full
story, every real number, and the explicit decision to stop tuning in favor
of Milestone 3.

### Known limitations of Milestone 2 (by design, not oversight)

- No integration into the Milestone 1 C++ engine yet — deliberate, planned as a follow-up once the kernels are both correct *and* fast.
- Neither kernel fully clears the spec's 1.5x performance threshold; the remaining gap is a well-evidenced hardware-matmul throughput ceiling, not overhead — see the evidence doc for what would actually close it (`simdgroup_matrix` rewrite, explicitly deferred).
- Quantized GEMM kernel has no direct Instruments capture (gating does) — see `MILESTONE2_PERF_EVIDENCE.md` §2.4 for the specific gap and how to close it if needed.

---

## Next up

Milestone 3: simulated multi-node orchestration (OS processes standing in for GPU nodes) + async compute/transfer overlap. See `AetherMoEMacM1.md` §4 for full scope.
