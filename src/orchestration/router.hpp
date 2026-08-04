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
#include <unordered_set>
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

    // ===================== Milestone 4: graceful degradation =====================
    //
    // route_batch (above) is deliberately left untouched -- Milestone 3's
    // KilledWorkerIsDetectedNotHung test depends on a single dead shard
    // throwing all the way out of that call, and that's still the right
    // contract for a caller that wants "fail the whole batch loudly."
    //
    // This method is for a caller that instead wants the control plane to
    // keep running through a node failure (the M4 spec's actual ask):
    // every token gets exactly one RoutedResult back -- either a genuine
    // result, or `failed == true` with a reason -- never a throw, and
    // never a silently missing batch_position. A shard that fails once is
    // remembered via `is_shard_unhealthy()` so later calls in the same
    // Router's lifetime don't keep paying the cost (and the latency) of
    // talking to a socket that's already known to be gone, and healthy
    // shards in the SAME batch are unaffected by another shard's failure.
    //
    // Deliberately NOT built: any reconnection/backoff/retry policy for an
    // unhealthy shard. The spec asks for isolating a fault so the rest of
    // the cluster keeps serving traffic, not a general node-recovery
    // system -- adding one here would be solving a problem the spec
    // doesn't pose, the same judgment call this file's header comment
    // already makes about cascading overflow. A caller that needs a shard
    // back after a restart constructs a fresh Router (or a future,
    // separately-designed hook clears unhealthy_shards_ explicitly).
    std::vector<RoutedResult> route_batch_tolerant(const std::vector<RoutedToken>& tokens) {
        std::vector<uint32_t> effective_expert(tokens.size());
        assign_effective_experts(tokens, effective_expert);

        std::unordered_map<uint32_t, std::vector<size_t>> indices_by_shard;
        for (size_t i = 0; i < tokens.size(); ++i) {
            indices_by_shard[shard_for_expert(effective_expert[i])].push_back(i);
        }

        std::vector<RoutedResult> ordered(tokens.size());
        std::vector<bool> filled(tokens.size(), false);

        // Every token in a failed/unhealthy group gets an explicit,
        // accounted-for failure result instead of being left as a missing
        // position -- this closure is the one place that invariant is
        // enforced, so every failure path below routes through it.
        auto fail_group = [&](const std::vector<size_t>& idxs, const std::string& reason) {
            for (size_t i : idxs) {
                RoutedResult r;
                r.token_id = tokens[i].token_id;
                r.batch_position = tokens[i].batch_position;
                r.effective_expert = effective_expert[i];
                r.failed = true;
                r.failure_reason = reason;
                ordered[r.batch_position] = std::move(r);
                filled[r.batch_position] = true;
            }
        };

        for (auto& [shard_id, idxs] : indices_by_shard) {
            if (unhealthy_shards_.count(shard_id)) {
                fail_group(idxs, "shard " + std::to_string(shard_id) +
                                      " marked unhealthy from a previous failure "
                                      "in this Router's lifetime");
                continue;
            }

            std::vector<RoutedToken> group;
            group.reserve(idxs.size());
            for (size_t i : idxs) group.push_back(tokens[i]);

            std::vector<RoutedResult> shard_results;
            bool ok = true;
            std::string reason;
            try {
                transport_for_shard(shard_id).send(encode_token_batch(group));
                auto raw = transport_for_shard(shard_id).receive();
                shard_results = decode_result_batch(raw);
            } catch (const std::exception& e) {
                ok = false;
                reason = e.what();
            } catch (...) {
                ok = false;
                reason = "unknown error communicating with shard " + std::to_string(shard_id);
            }

            if (!ok) {
                unhealthy_shards_.insert(shard_id);
                fail_group(idxs, reason);
                continue;
            }

            // Map results this shard actually returned back onto their
            // batch positions, carrying over this shard's per-token
            // effective_expert decision (looked up by batch_position,
            // since `ordered`/results are indexed by batch_position while
            // `effective_expert` is indexed by original-vector position --
            // conflating the two here was the actual bug caught while
            // self-testing this method, see MILESTONE4_PROGRESS.md).
            std::unordered_map<uint32_t, uint32_t> pos_to_effective;
            for (size_t i : idxs) pos_to_effective[tokens[i].batch_position] = effective_expert[i];

            for (auto& r : shard_results) {
                if (r.batch_position < filled.size() && !filled[r.batch_position]) {
                    auto eff_it = pos_to_effective.find(r.batch_position);
                    if (eff_it != pos_to_effective.end()) r.effective_expert = eff_it->second;
                    ordered[r.batch_position] = std::move(r);
                    filled[r.batch_position] = true;
                }
            }
            // Anything this shard was supposed to answer but didn't (a
            // partial/short response, not a hard transport exception) is
            // still marked failed rather than left as a missing position
            // -- same "accounted for, never silently dropped" contract as
            // every other path through this method.
            for (size_t i : idxs) {
                if (!filled[tokens[i].batch_position]) {
                    RoutedResult r;
                    r.token_id = tokens[i].token_id;
                    r.batch_position = tokens[i].batch_position;
                    r.effective_expert = effective_expert[i];
                    r.failed = true;
                    r.failure_reason = "missing result from shard " + std::to_string(shard_id);
                    ordered[r.batch_position] = std::move(r);
                    filled[r.batch_position] = true;
                }
            }
        }
        // Every position was touched by exactly one of the paths above
        // (a shard's success/failure, or the pre-known-unhealthy check),
        // since indices_by_shard partitions every input token exactly
        // once -- so `filled` is guaranteed all-true here. effective_expert
        // was already set correctly at each assignment site above (keyed
        // by batch_position, not blindly re-applied here by input-vector
        // index -- see the comment above about why that distinction
        // matters).
        return ordered;
    }

    bool is_shard_unhealthy(uint32_t shard_id) const {
        return unhealthy_shards_.count(shard_id) != 0;
    }

    size_t unhealthy_shard_count() const { return unhealthy_shards_.size(); }

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
    // Milestone 4: shards route_batch_tolerant has seen fail at least once.
    // NOT touched by route_batch -- that method's contract (throw on any
    // failure) is unaffected by this state entirely.
    std::unordered_set<uint32_t> unhealthy_shards_;
};

}