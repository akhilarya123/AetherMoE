// AetherMoE — src/orchestration/worker_shard.hpp
//
// The worker side of one simulated node: loops receiving token batches,
// runs a placeholder "expert compute" (no real model exists yet, same
// convention as Sequence's placeholder token ids in Milestone 1), and sends
// results back. Exits cleanly when the router's end of the transport closes
// (PeerClosedError on receive()) -- that's the normal, expected shutdown
// signal, not treated as an error.

#pragma once

#include "collective_transport.hpp"
#include "routed_token.hpp"
#include "serialization.hpp"

namespace aether::orchestration {

// Placeholder "expert compute": deterministic and easy to verify in tests
// (negate every element, tag with which shard did it) -- Milestone 3 is
// about proving tokens reach the RIGHT shard and come back in the RIGHT
// order, not about doing real expert math (that's Milestone 2's job,
// integrated later per the handover's own decision to keep that a separate
// step).
inline RoutedResult process_token_placeholder(const RoutedToken& t, uint32_t shard_id) {
    RoutedResult r;
    r.token_id = t.token_id;
    r.batch_position = t.batch_position;
    r.processed_by_shard = shard_id;
    r.payload.reserve(t.payload.size());
    for (float v : t.payload) {
        r.payload.push_back(-v);
    }
    return r;
}

// Runs until the transport's peer (the router) goes away -- either cleanly
// (PeerClosedError) or abruptly (any other transport-level exception, e.g.
// a connection reset if the router closes while this worker's reply to a
// PREVIOUS round is still unread in the socket buffer -- a real scenario
// found via fault-injection testing, not a hypothetical). From a worker's
// point of view, ANY error talking to its router means the same thing --
// the session is over, time to shut down -- not specifically "was it a
// clean close or not". Treating only PeerClosedError as the shutdown
// signal and letting everything else propagate meant an uncaught
// exception could escape this loop entirely on that path. Returns the
// number of batches processed (mainly for test assertions/logging).
inline size_t run_worker_loop(ICollectiveTransport& transport, uint32_t shard_id) {
    size_t batches_processed = 0;
    while (true) {
        std::vector<uint8_t> raw;
        try {
            raw = transport.receive();
        } catch (const std::exception&) {
            break;  // router is gone, cleanly or not -- clean shutdown either way
        }

        std::vector<RoutedToken> tokens = decode_token_batch(raw);
        std::vector<RoutedResult> results;
        results.reserve(tokens.size());
        for (const auto& t : tokens) {
            results.push_back(process_token_placeholder(t, shard_id));
        }
        try {
            transport.send(encode_result_batch(results));
        } catch (const std::exception&) {
            break;  // router disappeared between our receive() and this send()
        }
        ++batches_processed;
    }
    return batches_processed;
}

}  // namespace aether::orchestration
