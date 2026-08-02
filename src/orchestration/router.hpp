// AetherMoE — src/orchestration/router.hpp
//
// Scatter/gather across simulated expert shards. Given a batch of
// RoutedTokens (each already carrying its Milestone-2-gating-assigned
// primary_expert), groups tokens by destination shard, dispatches each
// group over that shard's transport, gathers the results back, and
// reassembles them into the ORIGINAL batch order -- shards process tokens
// in whatever order they arrive, batch_position is what makes "reassembled
// correctly" a checkable property rather than an assumption.
//
// Phase B: expert-capacity ceiling. If more than `expert_capacity` tokens
// in one batch choose the same primary_expert, the overflow (tokens beyond
// the first `expert_capacity`, ranked by batch_position so the decision is
// deterministic and independently reproducible in tests) gets rerouted to
// each overflowing token's OWN secondary_expert instead. This only changes
// which expert a token's shard is looked up by -- scatter, dispatch,
// gather, and reassembly are completely unchanged, and workers have zero
// concept of capacity; it's a router-only decision, applied before
// scatter and annotated onto results after gather.
//
// Deliberately NOT handled: cascading overflow (secondary_expert also over
// its own capacity). The spec asks for one hop -- "rerouted to their
// second-choice expert" -- not a general multi-level rebalancer, and
// adding cascading logic here would be solving a problem the spec doesn't
// pose. If secondary_expert is also hot, its tokens simply queue there;
// noted as a known, deliberate limitation rather than a gap discovered
// later.

#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "collective_transport.hpp"
#include "routed_token.hpp"
#include "serialization.hpp"

namespace aether::orchestration {

struct RouterConfig {
    // Max tokens per expert per batch before overflow reroutes to
    // secondary_expert. Default is "unlimited" (Phase A behavior,
    // unchanged) -- capacity ceiling is opt-in via this config, not a
    // silent behavior change for existing callers/tests.
    size_t expert_capacity = std::numeric_limits<size_t>::max();
};

class Router {
public:
    Router(std::unordered_map<uint32_t, uint32_t> expert_to_shard,
           std::unordered_map<uint32_t, ICollectiveTransport*> shard_transports,
           RouterConfig config = {})
        : expert_to_shard_(std::move(expert_to_shard)),
          shard_transports_(std::move(shard_transports)),
          config_(config) {}

    // Routes one full batch through to completion: apply the capacity
    // ceiling, scatter, dispatch, gather, reassemble. Throws if any shard's
    // results are missing, duplicated, or reference an out-of-range
    // batch_position -- silently returning a partial/wrong-order result
    // would defeat the entire point of this class.
    std::vector<RoutedResult> route_batch(const std::vector<RoutedToken>& tokens) {
        std::vector<uint32_t> effective_expert(tokens.size());
        assign_effective_experts(tokens, effective_expert);

        std::unordered_map<uint32_t, std::vector<RoutedToken>> by_shard;
        for (size_t i = 0; i < tokens.size(); ++i) {
            by_shard[shard_for_expert(effective_expert[i])].push_back(tokens[i]);
        }

        std::vector<uint32_t> shards_sent;
        shards_sent.reserve(by_shard.size());
        for (auto& [shard_id, group] : by_shard) {
            transport_for_shard(shard_id).send(encode_token_batch(group));
            shards_sent.push_back(shard_id);
        }

        std::vector<RoutedResult> flat_results;
        flat_results.reserve(tokens.size());
        for (uint32_t shard_id : shards_sent) {
            auto raw = transport_for_shard(shard_id).receive();
            for (auto& r : decode_result_batch(raw)) {
                flat_results.push_back(std::move(r));
            }
        }

        auto ordered = reassemble(tokens.size(), flat_results);
        // Annotate effective_expert onto results -- router-only bookkeeping,
        // never sent over the wire (see the field's doc comment in
        // routed_token.hpp).
        for (size_t i = 0; i < ordered.size(); ++i) {
            ordered[i].effective_expert = effective_expert[i];
        }
        return ordered;
    }

private:
    // Phase B: determines which expert each token is ACTUALLY assigned to
    // after applying the capacity ceiling. Tokens are ranked within their
    // primary_expert group by batch_position (not arrival/iteration order,
    // which would make this non-deterministic and untestable) -- the first
    // `expert_capacity` keep their primary_expert; the rest overflow to
    // their own secondary_expert.
    void assign_effective_experts(const std::vector<RoutedToken>& tokens,
                                    std::vector<uint32_t>& effective_expert) const {
        std::unordered_map<uint32_t, std::vector<size_t>> indices_by_primary;
        for (size_t i = 0; i < tokens.size(); ++i) {
            indices_by_primary[tokens[i].primary_expert].push_back(i);
        }
        for (auto& [expert, indices] : indices_by_primary) {
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                return tokens[a].batch_position < tokens[b].batch_position;
            });
            for (size_t rank = 0; rank < indices.size(); ++rank) {
                size_t idx = indices[rank];
                effective_expert[idx] = (rank < config_.expert_capacity)
                                             ? tokens[idx].primary_expert
                                             : tokens[idx].secondary_expert;  // OVERFLOW
            }
        }
    }

    uint32_t shard_for_expert(uint32_t expert_id) const {
        auto it = expert_to_shard_.find(expert_id);
        if (it == expert_to_shard_.end()) {
            throw std::runtime_error(
                "Router: no shard mapping for expert " + std::to_string(expert_id));
        }
        return it->second;
    }

    ICollectiveTransport& transport_for_shard(uint32_t shard_id) const {
        auto it = shard_transports_.find(shard_id);
        if (it == shard_transports_.end()) {
            throw std::runtime_error(
                "Router: no transport registered for shard " + std::to_string(shard_id));
        }
        return *it->second;
    }

    static std::vector<RoutedResult> reassemble(size_t original_batch_size,
                                                  std::vector<RoutedResult>& flat_results) {
        std::vector<RoutedResult> ordered(original_batch_size);
        std::vector<bool> filled(original_batch_size, false);

        for (auto& r : flat_results) {
            if (r.batch_position >= original_batch_size) {
                throw std::runtime_error(
                    "Router: result batch_position " + std::to_string(r.batch_position) +
                    " out of range for batch of size " + std::to_string(original_batch_size));
            }
            if (filled[r.batch_position]) {
                throw std::runtime_error(
                    "Router: duplicate result for batch_position " +
                    std::to_string(r.batch_position) + " (dropped/duplicated token bug)");
            }
            ordered[r.batch_position] = std::move(r);
            filled[r.batch_position] = true;
        }
        for (size_t i = 0; i < filled.size(); ++i) {
            if (!filled[i]) {
                throw std::runtime_error(
                    "Router: missing result for batch_position " + std::to_string(i) +
                    " (dropped token bug)");
            }
        }
        return ordered;
    }

    std::unordered_map<uint32_t, uint32_t> expert_to_shard_;
    std::unordered_map<uint32_t, ICollectiveTransport*> shard_transports_;
    RouterConfig config_;
};

}