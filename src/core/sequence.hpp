// AetherMoE — src/core/sequence.hpp
//
// Per-request state machine. A "sequence" is one in-flight generation
// request as it moves through the engine:
//
//   PREFILL  -> the prompt tokens are being processed (one or more chunks)
//   DECODE   -> generating new tokens one iteration at a time
//   FINISHED -> stop condition met (max_tokens reached here; a real engine
//               would also check EOS token / stop strings)
//
// This struct is intentionally dumb data + explicit transition methods
// rather than hiding state behind getters — the scheduler is the only thing
// that drives transitions, and tests should be able to construct/inspect a
// Sequence without going through the whole engine.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aether::core {

enum class SequencePhase : uint8_t {
    PREFILL,
    DECODE,
    FINISHED,
};

inline const char* to_string(SequencePhase p) {
    switch (p) {
        case SequencePhase::PREFILL: return "PREFILL";
        case SequencePhase::DECODE: return "DECODE";
        case SequencePhase::FINISHED: return "FINISHED";
    }
    return "UNKNOWN";
}

struct Sequence {
    uint64_t id;
    SequencePhase phase = SequencePhase::PREFILL;

    // Prompt / generation bookkeeping. token ids are placeholders (uint32)
    // since no tokenizer/model exists yet at Milestone 1 — the scheduler and
    // page table only need counts and block ownership, not real vocab ids.
    std::vector<uint32_t> prompt_tokens;
    std::vector<uint32_t> generated_tokens;
    uint32_t max_new_tokens = 0;

    // How many prompt tokens have been chunked through PREFILL so far.
    // Allows the scheduler to admit large prompts in slices rather than
    // requiring the whole prompt processed in one iteration.
    size_t prefill_cursor = 0;

    // Physical block ids owned by this sequence in the page table.
    std::vector<int> owned_blocks;

    explicit Sequence(uint64_t seq_id, std::vector<uint32_t> prompt,
                       uint32_t max_new)
        : id(seq_id),
          prompt_tokens(std::move(prompt)),
          max_new_tokens(max_new) {}

    bool prefill_complete() const {
        return prefill_cursor >= prompt_tokens.size();
    }

    size_t total_tokens() const {
        return prompt_tokens.size() + generated_tokens.size();
    }

    // Advances the state machine. Throws on illegal transitions so bugs in
    // the scheduler surface immediately in tests rather than silently
    // corrupting state.
    void transition_to(SequencePhase next) {
        static const bool legal[3][3] = {
            /*            PREFILL DECODE FINISHED */
            /* PREFILL  */ {true,  true,  true},
            /* DECODE   */ {false, true,  true},
            /* FINISHED */ {false, false, true},
        };
        auto from = static_cast<size_t>(phase);
        auto to = static_cast<size_t>(next);
        if (!legal[from][to]) {
            throw std::logic_error(
                std::string("illegal sequence transition: ") +
                to_string(phase) + " -> " + to_string(next));
        }
        phase = next;
    }
};

}  // namespace aether::core
