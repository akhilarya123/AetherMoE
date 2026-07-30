# AetherMoE — Milestone 2: Fused Gating & Quantized GEMM (MLX/Metal)

## Status — read this first

Unlike Milestone 1, **I could not run or test any of this code myself.** My
build environment has no GPU, no MLX, no Metal — that's a hard constraint,
not a shortcut I took. Here's exactly what level of confidence exists for
each piece:

| Component | Confidence | Why |
|---|---|---|
| `prototype/math_proto.py` | **Verified — I ran it** | Pure NumPy, proves the top-k/softmax and INT8 quantization math is correct, including the tie-breaking edge case, before any MLX code was written. |
| `tests/python/test_math_prototype.py` | **Verified — I ran it** | Same checks as above, as real pytest cases. 5/5 pass. |
| `src/kernels/reference_gating.py` | Likely correct | Uses only well-established MLX ops (matmul, argsort, softmax). Lowest risk of the MLX-dependent files. |
| `src/kernels/fused_gating_metal.py` | **Unverified — needs your Mac** | The Metal kernel logic mirrors the proven math, but the exact `mx.fast.metal_kernel(...)` call signature is the single highest-risk part of this milestone (see the HONESTY NOTE at the top of that file). |
| `src/kernels/quantized_gemm.py` | **Unverified — needs your Mac** | Same API-signature risk as above. |
| `validation/pytorch_reference.py` | Likely correct | Standard PyTorch ops; falls back to CPU if `mps` isn't available. |

**What to do when something breaks:** paste me the exact traceback. If the
error is inside the Metal kernel source (a compile error mentioning line
numbers in a `.metal`-like context), that's a logic bug in `_KERNEL_SOURCE`.
If the error is a Python `TypeError`/`AttributeError` on the `kernel(...)`
call itself, that's almost certainly the `mx.fast.metal_kernel` API shape
not matching what I guessed — run:
```bash
python3 -c "import mlx.core as mx; help(mx.fast.metal_kernel)"
```
and send me that output; I'll fix the call to match your installed version.

## Setup

```bash
cd AetherMoE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-m2.txt
```

`mlx` will only install/work on Apple Silicon. `torch` should pick up the
`mps` backend automatically on macOS 12.3+.

## Run things in this order

**1. The dependency-free sanity gate (should just work, proves the
environment itself is fine before touching MLX):**
```bash
pytest tests/python/test_math_prototype.py -v
```

**2. The MLX reference implementation (Tier A — lower risk):**
```bash
python3 -c "
import mlx.core as mx
from src.kernels.reference_gating import reference_gating
import numpy as np
rng = np.random.default_rng(0)
h = mx.array(rng.normal(size=(4,16)).astype(np.float32))
w = mx.array(rng.normal(size=(16,8)).astype(np.float32))
b = mx.array(np.zeros(8, dtype=np.float32))
idx, val = reference_gating(h, w, b, k=3)
print(idx)
print(val)
print('row sums:', np.array(val).sum(axis=-1))
"
```
If this errors, it's likely `mx.take_along_axis` not existing in your MLX
version — the fallback is commented directly below the line that uses it
in `reference_gating.py`.

**3. The fused Metal kernel (Tier B — the actual deliverable, most likely
to need iteration):**
```bash
pytest tests/python/test_gating_numerical.py -v
```

**4. Property-based tests (once #3 passes):**
```bash
pytest tests/python/test_gating_property.py -v
```

**5. Quantized GEMM:**
```bash
pytest tests/python/test_quantized_gemm.py -v
```

**6. Performance regression benchmark:**
```bash
python3 benchmarks/bench_gating.py
python3 benchmarks/bench_gating.py --n-tokens 512 --d-model 4096 --num-experts 64 --k 8
```

**7. Run everything:**
```bash
pytest tests/python/ -v
```

## Profiling (do this after correctness passes, not before)

Once tests are green, capture an Instruments Metal System Trace comparing
`reference_gating` vs `fused_gating` under the same shape — the Milestone 2
Definition of Done requires evidence of reduced memory round-trips, not
just a claim. In Xcode: Product menu → Profile, or launch Instruments
directly and attach to the running `python3` process while it loops the
benchmark. We can go through this together once the kernel itself is
working — no point profiling code that isn't correct yet.

## What's NOT done yet (intentionally, per the milestone scope)

- No integration back into the Milestone 1 C++ engine yet — that's a
  deliberate follow-up once the kernels are proven correct in isolation,
  not an oversight.
- The quantized GEMM kernel is one-thread-per-output-element, not
  tiled/blocked — correct but not performance-competitive with MLX's
  built-in matmul at large sizes. That's flagged as a known follow-up in
  the kernel's own docstring, not a bug to chase right now.
- Instruments profiling evidence — needs your hands-on-keyboard time in
  Xcode, can't be scripted from here.

## Next: once you've run this

Send me:
1. Output of `pytest tests/python/ -v` (full output, especially any
   failures/tracebacks)
2. Output of `python3 benchmarks/bench_gating.py`
3. If step 2/3 above errored, the `help(mx.fast.metal_kernel)` output

Then we'll fix whatever needs fixing and move to Milestone 3 (simulated
multi-node orchestration + async overlap).
