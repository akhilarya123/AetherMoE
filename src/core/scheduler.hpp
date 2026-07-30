// AetherMoE — src/core/scheduler.hpp
//
// Iteration-level ("continuous batching") scheduler. Unlike request-level
// batching — where a fixed set of requests runs together until ALL of them
// finish, wasting compute on padding for short sequences — this scheduler
// re-forms its batch every iteration, mixing:
//   - PREFILL work for newly-admitted (or partially-prefilled) sequences
//   - one DECODE step each for sequences already generating
// under a single token budget per iteration (max_batch_tokens_).
//
// This class does not do any GPU/model compute — that's Milestones 2+. What
// it owns is the *admission and bookkeeping* logic: which sequences run this
// iteration, how many prompt tokens of each get prefilled, page table growth,
// and phase transitions. A real "run the model on this batch" call is a
// pluggable callback so this scheduler is testable in isolation without a
// GPU, and Milestone 2's kernels slot in later without touching this file.

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "page_table.hpp"
#include "sequence.hpp"

namespace aether::core {

// Describes what happened to one sequence during a single iteration, for
// telemetry/testing.
struct StepResult {
    uint64_t seq_id;
    bool was_prefill_chunk;
    size_t tokens_processed;  // prompt tokens (prefill) or 1 (decode)
    SequencePhase phase_after;
};

struct SchedulerConfig {
    size_t max_batch_tokens = 2048;  // token budget per iteration
    size_t max_prefill_chunk = 512;  // largest prefill slice per seq/iteration
};

class ContinuousBatchingScheduler {
public:
    ContinuousBatchingScheduler(SchedulerConfig config,
                                 PagedKVCacheAllocator& page_table)
        : config_(config), page_table_(page_table) {}

    // Admits a new sequence into the waiting queue. Thread-unsafe by design:
    // callers are expected to enqueue via the SPMC ring buffer and have a
    // single scheduler-owning thread call admit()/step() (see main.cpp).
    void admit(std::shared_ptr<Sequence> seq) {
        waiting_.push_back(std::move(seq));
    }

    size_t num_waiting() const { return waiting_.size(); }
    size_t num_active() const { return active_.size(); }

    // Runs one scheduling iteration. Returns per-sequence results for this
    // iteration (empty if there was no work at all).
    std::vector<StepResult> step() {
        std::vector<StepResult> results;
        size_t budget = config_.max_batch_tokens;

        // 1. Decode step for every currently-active DECODE sequence first —
        //    in-flight generations get priority so they're never starved by
        //    a stream of new prefill admissions (this is the guarantee
        //    tested by "no disruption to in-flight decode" in the M1 DoD).
        for (auto it = active_.begin(); it != active_.end() && budget > 0;) {
            auto& seq = *it;
            if (seq->phase == SequencePhase::DECODE) {
                if (!page_table_.ensure_capacity(seq->id, seq->total_tokens() + 1)) {
                    ++it;  // OOM this iteration; try again next iteration
                    continue;
                }
                seq->generated_tokens.push_back(kPlaceholderToken);
                budget -= 1;
                bool finished = seq->generated_tokens.size() >= seq->max_new_tokens;
                if (finished) {
                    seq->transition_to(SequencePhase::FINISHED);
                    page_table_.free_sequence(seq->id);
                }
                results.push_back({seq->id, false, 1, seq->phase});
                if (finished) {
                    it = active_.erase(it);
                    continue;
                }
            }
            ++it;
        }

        // 2. Spend remaining budget admitting/advancing PREFILL work, in
        //    FIFO order over the waiting queue plus any active sequences
        //    still mid-prefill (large prompts spanning multiple iterations).
        for (auto it = active_.begin(); it != active_.end() && budget > 0; ++it) {
            auto& seq = *it;
            if (seq->phase != SequencePhase::PREFILL) continue;
            size_t chunk = do_prefill_chunk(*seq, budget);
            if (chunk > 0) {
                budget -= chunk;
                results.push_back({seq->id, true, chunk, seq->phase});
            }
        }

        while (budget > 0 && !waiting_.empty()) {
            auto seq = waiting_.front();
            size_t chunk = do_prefill_chunk(*seq, budget);
            if (chunk == 0) break;  // no budget or no capacity — stop admitting
            budget -= chunk;
            results.push_back({seq->id, true, chunk, seq->phase});
            waiting_.pop_front();
            active_.push_back(seq);
        }

        return results;
    }

private:
    static constexpr uint32_t kPlaceholderToken = 0xFFFFFFFF;

    // Prefills up to max_prefill_chunk tokens (bounded further by remaining
    // iteration budget) for one sequence. Returns tokens actually processed
    // (0 if blocked on page-table capacity or out of budget).
    size_t do_prefill_chunk(Sequence& seq, size_t budget) {
        if (seq.prefill_complete()) {
            seq.transition_to(SequencePhase::DECODE);
            return 0;
        }
        size_t remaining_prompt = seq.prompt_tokens.size() - seq.prefill_cursor;
        size_t chunk = std::min({remaining_prompt, config_.max_prefill_chunk, budget});
        if (chunk == 0) return 0;

        if (!page_table_.ensure_capacity(seq.id, seq.prefill_cursor + chunk)) {
            return 0;  // page table OOM — retry next iteration
        }
        seq.prefill_cursor += chunk;
        if (seq.prefill_complete()) {
            seq.transition_to(SequencePhase::DECODE);
        }
        return chunk;
    }

    SchedulerConfig config_;
    PagedKVCacheAllocator& page_table_;
    std::deque<std::shared_ptr<Sequence>> waiting_;
    std::deque<std::shared_ptr<Sequence>> active_;
};

}  // namespace aether::core
