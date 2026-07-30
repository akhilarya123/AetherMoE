// AetherMoE — src/main.cpp
//
// Wires the Milestone 1 pieces together into a runnable (if not yet
// model-backed) engine:
//
//   [HTTP client] --POST /generate--> [IngressServer thread]
//                                          |
//                                   try_push(Sequence)
//                                          v
//                              [LockFreeRingBuffer<Sequence>]
//                                          |
//                                   try_pop (drain loop)
//                                          v
//                        [ContinuousBatchingScheduler.step()]
//                                          |
//                               updates SequenceStatusTable
//                                          v
//                       [HTTP client] <--GET /status/:id--
//
// There is no model here yet (that's Milestone 2's MoE kernels running on
// MLX/Metal) — decode steps just append a placeholder token so the whole
// pipeline (admission, paging, phase transitions, backpressure) can be load
// tested end-to-end before any GPU code exists.

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

#include "api/ingress_server.hpp"
#include "core/page_table.hpp"
#include "core/scheduler.hpp"

namespace {
std::atomic<bool> g_shutdown{false};
void handle_sigint(int) { g_shutdown.store(true, std::memory_order_release); }
}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    int port = 8080;
    if (argc > 1) port = std::atoi(argv[1]);

    // --- Config ---
    // These sizes are deliberately small placeholder defaults for a
    // laptop-scale Milestone 1 smoke test; Milestone 2+ will size the page
    // table against real KV-cache memory budgets once tensor shapes exist.
    constexpr size_t kBlockSizeTokens = 16;
    constexpr size_t kNumPhysicalBlocks = 4096;  // 4096*16 = 65536 token-slots

    aether::core::PagedKVCacheAllocator page_table(kBlockSizeTokens,
                                                      kNumPhysicalBlocks);
    aether::core::SchedulerConfig sched_cfg;
    sched_cfg.max_batch_tokens = 2048;
    sched_cfg.max_prefill_chunk = 512;
    aether::core::ContinuousBatchingScheduler scheduler(sched_cfg, page_table);

    aether::api::IngressQueue queue;
    aether::api::SequenceStatusTable status_table;
    std::atomic<uint64_t> next_seq_id{1};

    aether::api::IngressServer server(queue, status_table, next_seq_id);
    std::thread server_thread([&] { server.run("0.0.0.0", port); });

    std::cout << "[engine] scheduler loop starting (block_size=" << kBlockSizeTokens
              << " tokens, physical_blocks=" << kNumPhysicalBlocks << ")\n";

    // Scheduler / drain loop: pulls newly-admitted sequences off the ring
    // buffer, then iterates the scheduler. Runs on its own thread so the
    // HTTP thread pool never blocks on scheduling work.
    while (!g_shutdown.load(std::memory_order_acquire)) {
        std::shared_ptr<aether::core::Sequence> seq;
        int drained = 0;
        while (drained < 256 && queue.try_pop(seq)) {
            scheduler.admit(seq);
            ++drained;
        }

        auto results = scheduler.step();
        for (auto& r : results) {
            status_table.upsert(r.seq_id, r.phase_after, /*generated_count=*/0);
        }

        if (drained == 0 && results.empty()) {
            // No work at all this tick — avoid a busy-spin burning a full
            // CPU core for nothing.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    std::cout << "[engine] shutdown signal received, exiting\n";
    server_thread.detach();  // httplib has no clean async stop() wired here yet
    return 0;
}
