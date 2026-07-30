# AetherMoE — Milestone 1: Foundation & Core Engine Loop

Status: built and verified (functional tests + ThreadSanitizer clean) on a
Linux sandbox. **Not yet verified on your M1** — that's the next step, and
the one part I genuinely can't do for you since I don't have Metal/Apple
Silicon access. Everything below is plain C++20 (no Metal/MLX yet — that
starts in Milestone 2), so it should build and run as-is on macOS.

## What's in this milestone

| Component | File | What it does |
|---|---|---|
| Lock-free SPMC ring buffer | `src/core/spmc_ring_buffer.hpp` | Vyukov bounded-queue algorithm. Ingress thread pushes, scheduler thread pops. |
| Sequence state machine | `src/core/sequence.hpp` | PREFILL → DECODE → FINISHED, with illegal-transition guards. |
| Paged KV-cache allocator | `src/core/page_table.hpp` | Block-based virtual→physical allocator (PagedAttention-style), lazy growth, O(1) free-list. |
| Continuous batching scheduler | `src/core/scheduler.hpp` | Iteration-level scheduler: decode steps prioritized over new prefill admission, chunked prefill for long prompts. |
| Ingress HTTP API | `src/api/ingress_server.{hpp,cpp}` | `POST /generate`, `GET /status/:id`, `GET /healthz`, via cpp-httplib. |
| Engine entrypoint | `src/main.cpp` | Wires everything together into one runnable binary. |
| Load generator | `benchmarks/load_generator.cpp` | Concurrent client for the throughput/backpressure test. |
| Tests | `tests/*.cpp` | 18 GTest cases: FIFO correctness, wraparound, SPMC stress (200k ops / 8 consumers), page-table exhaustion/fragmentation/concurrency, scheduler admission + priority + OOM-blocking + randomized-workload soak. |

Everything is dependency-free except three header-only libraries fetched
automatically by CMake the first time you configure: GoogleTest, cpp-httplib,
and nlohmann/json. No Python, no Docker, no paid services — $0 as promised.

## One-time Mac setup

Open Terminal and run:

```bash
# 1. Xcode Command Line Tools (gives you clang, make, git)
xcode-select --install
# A popup will appear — click Install and wait for it to finish (~5-10 min).
# If you already have it, this will just print an error saying so — that's fine.

# 2. Homebrew (macOS package manager) — skip if you already have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Follow the on-screen instructions at the end to add brew to your PATH,
# then close and reopen Terminal (or run the `eval "$(...)"` line it prints).

# 3. CMake
brew install cmake

# 4. Verify
clang++ --version   # should show Apple clang, e.g. "Apple clang version 15..."
cmake --version     # should show 3.20 or higher
```

## Build & test

```bash
cd AetherMoE
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
```

The first `cmake -B build` will take a minute or two — it's downloading
GoogleTest/cpp-httplib/nlohmann-json from GitHub over your internet
connection (one-time; cached in `build/_deps` afterward).

Run the unit test suite:

```bash
./build/aether_tests
```

You should see `[PASSED] 18 tests`. If anything fails on your machine that
passed in my sandbox, that's genuinely useful signal (possible macOS/ARM
atomics or timing difference) — paste me the output and we'll dig in.

### Concurrency check (recommended, catches lock-free bugs sanitizers are
built to find — I already ran this clean on Linux, but Apple's TSan
implementation can surface different things):

```bash
cmake -B build-tsan -DAETHER_SANITIZE=thread
cmake --build build-tsan -j$(sysctl -n hw.ncpu) --target aether_tests
./build-tsan/aether_tests
```

Should print `[PASSED] 18 tests` with no `WARNING: ThreadSanitizer` blocks.

## Run the engine

```bash
./build/aether_engine 8080
```

In another terminal:

```bash
curl http://127.0.0.1:8080/healthz
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt_tokens":[1,2,3,4,5],"max_new_tokens":5}'
# -> {"accepted":true,"seq_id":1}

curl http://127.0.0.1:8080/status/1
# -> {"generated_count":0,"phase":"FINISHED","seq_id":1}
```

There's no real model yet — decode steps append a placeholder token, which
is why a 5-token request finishes near-instantly. That's intentional: this
milestone proves the ingress/scheduling/paging pipeline end-to-end before
any GPU compute exists.

## The load test — please run this and report back the numbers

This is the one I most want your real hardware's numbers on:

```bash
./build/load_generator 127.0.0.1 8080 150 300
```

(150 concurrent clients × 300 requests = 45,000 requests, randomized 10-1000
token prompts, per the Milestone 1 spec.) Watch for:
- **accepted + backpressure(503) should equal total requests** — `errors`
  should be 0. A nonzero `errors` count means connections are being dropped,
  not gracefully rejected.
- **throughput (req/s)** — the milestone target is 10,000+ req/s. On my
  1-core Linux sandbox I measured ~900 req/s with zero errors and zero
  drops, which tells me the *pipeline is correct* but says nothing about
  whether the *number* holds up on real multi-core hardware. Please share
  what you get.

If throughput falls well short of 10k on your M1, the likely culprit is
cpp-httplib's thread-per-connection model rather than anything in the
ring buffer/scheduler — I already found and fixed one real bug this way
(the library's default connection backlog was only 5, which was silently
dropping connections under load; it's now 1024). If we need to go faster
still, the next lever is replacing cpp-httplib with a leaner event-loop
based HTTP layer — but let's see your numbers first before deciding that's
necessary.

## Known limitations of this milestone (by design, not oversight)

- No tokenizer, no model, no MLX/Metal — placeholder tokens only. That's
  Milestone 2.
- Page table uses a mutex, not a lock-free free-list. Documented as a
  possible stretch goal in `page_table.hpp` — allocation happens once per
  `BLOCK_SIZE` tokens per sequence, not once per token, so this is unlikely
  to be the bottleneck, but worth revisiting if profiling says otherwise.
- `IngressServer::stop()` exists but cpp-httplib doesn't have a clean
  async-stop wired to it yet — Ctrl+C currently exits the scheduler loop
  cleanly but detaches the HTTP thread. Fine for a dev binary; would want
  fixing before anything resembling "production."

## Next up: Milestone 2

Whenever you're ready, say the word and we'll move on to the MoE model
architecture and the first MLX/Metal kernels — that's where your M1's GPU
actually gets used for the first time.
