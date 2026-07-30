// AetherMoE — src/api/ingress_server.hpp
//
// Thin async-facing HTTP ingress. Its only job is: accept a generation
// request over HTTP, turn it into a Sequence, and push it onto the SPMC ring
// buffer for the scheduler thread to pick up. It does NOT run any model
// code itself — that decoupling is the whole point of the ring buffer:
// the HTTP thread pool (I/O-bound) never blocks on the scheduler thread
// (compute-bound), and vice versa.
//
// Endpoints:
//   POST /generate   { "prompt_tokens": [int...], "max_new_tokens": int }
//                     -> { "seq_id": int, "accepted": true }
//                     or { "accepted": false, "reason": "..." } if the ring
//                     buffer is full (backpressure).
//   GET  /status/:id  -> { "seq_id": int, "phase": "PREFILL|DECODE|FINISHED",
//                          "generated_tokens": [int...] }
//   GET  /healthz     -> 200 OK once the engine is accepting traffic.
//
// Milestone 1 has no tokenizer or model yet, so /generate takes token ids
// directly (uint32 placeholders) rather than raw text — that seam is where
// Milestone 2's tokenizer will plug in without changing this file's
// public interface.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "core/sequence.hpp"
#include "core/spmc_ring_buffer.hpp"

namespace aether::api {

// Shared, thread-safe view of sequence status for the /status endpoint.
// The scheduler thread updates this after every step(); HTTP handler
// threads only read it. Guarded by a mutex since read/write frequency here
// is far lower than the ring buffer's hot path.
class SequenceStatusTable {
public:
    void upsert(uint64_t seq_id, core::SequencePhase phase,
                size_t generated_count) {
        std::lock_guard<std::mutex> lock(mu_);
        table_[seq_id] = {phase, generated_count};
    }

    struct Status {
        core::SequencePhase phase;
        size_t generated_count;
    };

    bool get(uint64_t seq_id, Status& out) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = table_.find(seq_id);
        if (it == table_.end()) return false;
        out = it->second;
        return true;
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, Status> table_;
};

constexpr size_t kIngressQueueCapacity = 4096;
using IngressQueue =
    core::LockFreeRingBuffer<std::shared_ptr<core::Sequence>, kIngressQueueCapacity>;

// Runs the HTTP server on the given port until stop() is called. Blocking —
// call from a dedicated thread (see main.cpp).
class IngressServer {
public:
    IngressServer(IngressQueue& queue, SequenceStatusTable& status_table,
                  std::atomic<uint64_t>& next_seq_id);

    void run(const std::string& host, int port);
    void stop();

private:
    IngressQueue& queue_;
    SequenceStatusTable& status_table_;
    std::atomic<uint64_t>& next_seq_id_;
    std::atomic<bool> stop_requested_{false};
};

}  // namespace aether::api
