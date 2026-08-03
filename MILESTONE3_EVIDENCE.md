# Milestone 3 — Completion Evidence

**Status: complete.** All phases built, tested, and evidenced with real data
from actual runs — not assumptions.

## Phase A — Transport + scatter/gather (`src/orchestration/`)

`ICollectiveTransport` interface, `UnixSocketTransport` (real `socketpair()`
IPC between real forked OS processes), binary message framing, `Router`
(scatter by expert->shard mapping, dispatch, gather, strict in-order
reassembly with duplicate/missing-result detection). Self-verified
extensively (own dependency-free harness, no network access for the real
GTest/CMake toolchain in this sandbox) before reaching the real build:
299,149+ assertions across 6 test cases, clean under
`-Wall -Wextra -Werror -fsanitize=address,undefined`, zero leaked/zombie
processes across 10+ independent randomized-seed runs. Confirmed passing
on the real Mac build (`./build/aether_tests`).

## Phase B — Expert-capacity ceiling

`RouterConfig::expert_capacity` (default unlimited, opt-in), overflow
tokens beyond capacity rerouted to their own `secondary_expert`, ranked
deterministically by `batch_position`. `RoutedResult::effective_expert`
added so tests (and future telemetry) can see the actual routing decision.
No changes needed to transport/serialization/worker loop — purely a
router-side seam, as designed in Phase A. Verified against an
independently-reimplemented expected-assignment calculation (not just the
router checking its own logic), plus a load-imbalance stress test with a
wall-clock no-stall bound. Confirmed passing on the real Mac build.

## Phase D — Fault injection (detection only — full recovery is explicitly Milestone 4's job per spec)

`KilledWorkerIsDetectedNotHung` test: real `SIGKILL` sent to a live worker
process mid-session, confirms the router detects the failure (throws)
within a bounded time rather than hanging. Two real bugs found and fixed
via this test, both worth knowing about:
1. **`SIGPIPE` killed the process** before any exception-handling code
   could run (writing to a dead peer's socket raises `SIGPIPE`, default
   disposition terminates the process). Fixed with
   `signal(SIGPIPE, SIG_IGN)` — deliberately NOT Linux's `MSG_NOSIGNAL`
   shortcut, since it doesn't exist on macOS, this project's actual target.
2. **Worker exception handling was too narrow** — only a clean peer-close
   was treated as the shutdown signal; a genuine connection-reset (a plain
   exception, not `PeerClosedError`) could escape the worker loop
   entirely. Broadened to treat any transport error as "router is gone,
   shut down", plus a defensive `try/catch(...)` around the worker
   callback in `spawn_workers()` so no exception can silently escape a
   forked child regardless of what the callback does.

**Note on section ordering:** Phase D (fault injection) was actually built
chronologically BEFORE Phase C (Metal double-buffering) — sections here
are ordered A/B/C/D for clarity on re-read, not development order. All
four are complete either way.

## Phase C — Double-buffered Metal execution (`tools/double_buffer_demo.mm`)

Two independent `MTLCommandQueue`s (transfer via blit, compute via a
placeholder kernel), synchronized entirely GPU-side via one
`MTLSharedEvent` with a monotonic counter — no CPU-side blocking waits
between slices (the only `waitUntilCompleted` in the program is a single
call on the final command buffer, purely so the program knows when to
read back results and exit).

**Correctness:** verified by reading back all output and comparing
against a CPU-computed reference after the final sync — passed at both
4MB/slice and 128MB/slice.

**Overlap evidence (the actual Definition-of-Done requirement) — real
Instruments Metal System Trace data, not assumed:**

| Pair | Overlap | Transfer start delay after compute began |
|---|---|---|
| compute[1] vs transfer[2] | 389µs | 7.8µs |
| compute[2] vs transfer[3] | 244µs | 8.0µs |
| compute[3] vs transfer[4] | 256µs | 7.7µs |
| compute[4] vs transfer[5] | 195µs | 9.8µs |
| compute[5] vs transfer[6] | 164µs | 35.4µs |
| compute[6] vs transfer[7] | 199µs | 6.5µs |

From iteration 1 onward, every transfer starts within 6-36µs of the
previous compute beginning and overlaps it substantially — confirmed via
the `metal-gpu-intervals` table (the actual GPU execution timeline).

**Methodology note worth keeping around:** an earlier pass at this
analysis used `metal-application-encoders-list` /
`metal-application-command-buffer-submissions`, which measure **CPU-side
command encoding order** — inherently sequential regardless of real GPU
overlap, since one CPU thread encodes commands one after another by
definition. That table showed zero overlap and a total span exceeding the
sum of individual durations, which looked like a genuine negative result
at the time. Only `metal-gpu-intervals` (the real GPU execution timeline)
revealed the actual overlap. Worth remembering for any future Metal
profiling in this project: encoding-order tables answer "when did the CPU
record this", not "when did the GPU actually run this" — a materially
different question.

**One anomaly, not a concern:** `compute[0]` starts ~15ms after
`transfer[1]` already completed — almost certainly one-time
pipeline/shader compilation warmup on the very first dispatch, a known
Metal phenomenon. Steady-state behavior (iteration 1 onward) is what
demonstrates the actual double-buffering mechanism.

## What's deliberately NOT done (Milestone 4's job per spec)

- Graceful degradation/recovery after a node failure (this milestone only
  proves *detection*).
- Any load-balancing beyond the single-hop capacity-ceiling reroute.
- Telemetry/benchmarking suite, chaos testing beyond the one fault-injection
  test here.
