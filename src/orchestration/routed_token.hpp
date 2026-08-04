// AetherMoE — src/orchestration/routed_token.hpp
//
// A RoutedToken is the unit of work moving through Milestone 3's simulated
// multi-node routing layer. It is deliberately NOT a Sequence (src/core) --
// Milestone 1 schedules whole in-flight *requests*; Milestone 3 routes
// individual *tokens* after Milestone 2's gating kernel has already decided
// which expert(s) each token wants. One Sequence's tokens can legitimately
// scatter across many different RoutedTokens/shards.
//
// `payload` stands in for a real hidden-state activation vector -- same
// spirit as Sequence's placeholder uint32 token ids in Milestone 1: no real
// model exists yet, so the router and transport only need something with a
// verifiable identity and enough bytes to prove real data moved intact
// across a real process boundary, not a specific numerical meaning.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aether::orchestration {

struct RoutedToken {
    uint64_t token_id;        // globally unique within one routing batch
    uint32_t batch_position;  // index in the ORIGINAL batch order -- this is
                               // what makes correct reassembly checkable;
                               // shards process tokens in whatever order
                               // they arrive, batch_position is how the
                               // router puts them back
    uint32_t primary_expert;   // expert id chosen by Milestone 2's gating kernel
    uint32_t secondary_expert; // second-choice expert, used only if primary's
                                // shard is over capacity (Phase B)
    std::vector<float> payload; // placeholder "hidden state"

    RoutedToken() = default;
    RoutedToken(uint64_t id, uint32_t pos, uint32_t primary, uint32_t secondary,
                std::vector<float> data)
        : token_id(id), batch_position(pos), primary_expert(primary),
          secondary_expert(secondary), payload(std::move(data)) {}
};

// A processed token coming BACK from a worker shard. Deliberately a
// separate type from RoutedToken (not just "reuse RoutedToken with payload
// overwritten") so it's structurally impossible to accidentally confuse an
// unprocessed request with a processed result in the same code path --
// caught this class of mistake in Milestone 1's own tests once already
// (see HANDOVER.md bug #3); cheap to design around it here from the start.
struct RoutedResult {
    uint64_t token_id;
    uint32_t batch_position;
    uint32_t processed_by_shard; // which shard actually computed this --
                                   // lets correctness tests verify routing,
                                   // not just reassembly
    // Phase B: which expert this token was ACTUALLY assigned to after the
    // capacity ceiling is applied -- primary_expert normally, secondary_expert
    // if it overflowed. Filled in by Router itself after gather (see
    // router.hpp), NOT sent over the wire -- workers have no concept of
    // capacity at all, this is a pure routing decision. Default-valued (0)
    // on any RoutedResult that hasn't gone through Router::route_batch yet.
    uint32_t effective_expert = 0;
    std::vector<float> payload;

    // Milestone 4: filled in by Router::route_batch_tolerant (never by
    // route_batch, and never sent over the wire -- same convention as
    // effective_expert above) when this token's shard failed instead of
    // returning a result. This is what makes "cleanly failed, never
    // silently dropped" checkable by a test: every RoutedResult in a
    // tolerant-mode batch is either a real result (failed == false) or an
    // accounted-for failure with a reason, there's no third, missing case.
    bool failed = false;
    std::string failure_reason;
};

}