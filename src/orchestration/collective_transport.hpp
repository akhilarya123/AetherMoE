// AetherMoE — src/orchestration/collective_transport.hpp
//
// ICollectiveTransport abstracts "how bytes move between the router and one
// worker shard" so the rest of the orchestration layer never depends on the
// specific IPC mechanism. Milestone 3 implements this once with Unix domain
// sockets (unix_socket_transport.hpp) as a stand-in for a real
// collective-communication library (NCCL-style All-to-All); a real backend
// could later satisfy the same interface without the router or scheduler
// changing at all -- same "wrap it behind a small interface" discipline the
// spec calls for explicitly.
//
// This is intentionally point-to-point (one transport instance per
// router<->shard edge), not a broadcast/all-reduce abstraction -- Milestone
// 3's scatter/gather is naturally point-to-point (each shard only cares
// about its own tokens), and the spec asks us to cover All-to-All/All-Reduce
// at the DESIGN level, not to build a general collective-ops library that
// nothing here would exercise.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace aether::orchestration {

// Thrown when the peer has cleanly closed the connection (e.g. a worker
// process exited) -- distinct from other IO errors so fault-injection tests
// (Phase D) can specifically detect "the node died" rather than treating
// every transport failure the same way.
class PeerClosedError : public std::runtime_error {
public:
    PeerClosedError() : std::runtime_error("transport peer closed the connection") {}
};

class ICollectiveTransport {
public:
    virtual ~ICollectiveTransport() = default;

    // Sends one length-framed message. Blocks until the full message is
    // written (or throws). Implementations must handle partial writes and
    // EINTR internally -- callers should never need to retry a send().
    virtual void send(const std::vector<uint8_t>& message) = 0;

    // Blocks until one full length-framed message has been received.
    // Throws PeerClosedError if the peer closed before sending anything
    // (a clean, expected "no more messages" signal, not necessarily an
    // error at the router level -- see Router::gather).
    virtual std::vector<uint8_t> receive() = 0;
};

}  // namespace aether::orchestration
