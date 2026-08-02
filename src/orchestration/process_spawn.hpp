// AetherMoE — src/orchestration/process_spawn.hpp
//
// Spawns N real worker OS processes (fork), each with its own end of a
// socketpair() created BEFORE any forking happens. This two-phase
// create-all-pairs-then-fork-all order (rather than create-pair/fork/
// create-next-pair/fork/...) matters for correctness, not just style: if
// pairs were created interleaved with forking, each child would inherit
// open fds for every OTHER shard's pair created before its own fork() call.
// Those leaked fds keep the pipe "alive" from the OS's point of view even
// after the REAL owning process closes its end, silently breaking EOF
// detection and causing exactly the kind of hang the spec's
// "concurrency/deadlock tests" are meant to catch. Creating everything
// first, then having each child close every fd except its own one, avoids
// the whole problem rather than needing a test to catch it later.
//
// Children exit via _exit(), not exit()/return-from-main -- `exit()` would
// run the PARENT process's static destructors and atexit handlers a second
// time (e.g. double-flushing stdio buffers), a classic, well-known fork()
// pitfall worth avoiding deliberately rather than discovering it later.

#pragma once

#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "unix_socket_transport.hpp"

namespace aether::orchestration {

struct SpawnedWorkers {
    // shard_id -> transport connected to that shard, owned by the router
    // (this) process.
    std::unordered_map<uint32_t, std::unique_ptr<UnixSocketTransport>> transports;
    std::vector<pid_t> child_pids;

    // Waits for every child to exit (call after the router side has closed
    // its transports, which signals each worker to shut down via
    // PeerClosedError -- see worker_shard.hpp). Returns exit statuses.
    std::vector<int> join_all() {
        std::vector<int> statuses;
        statuses.reserve(child_pids.size());
        for (pid_t pid : child_pids) {
            int status = 0;
            if (waitpid(pid, &status, 0) < 0) {
                throw std::runtime_error("process_spawn: waitpid failed");
            }
            statuses.push_back(status);
        }
        return statuses;
    }
};

// worker_fn runs INSIDE each forked child, given its own transport and
// shard_id; the child process _exit()s with its return value once
// worker_fn returns.
inline SpawnedWorkers spawn_workers(
    const std::vector<uint32_t>& shard_ids,
    const std::function<int(ICollectiveTransport&, uint32_t)>& worker_fn) {

    // Phase 1: create every socketpair up front, before any forking.
    struct Pair { int router_fd; int worker_fd; };
    std::vector<Pair> pairs;
    pairs.reserve(shard_ids.size());
    for (size_t i = 0; i < shard_ids.size(); ++i) {
        int fds[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
            throw std::runtime_error("process_spawn: socketpair() failed");
        }
        pairs.push_back({fds[0], fds[1]});
    }

    SpawnedWorkers result;

    // Phase 2: fork each child. Each child closes EVERY fd from EVERY pair
    // except its own worker_fd, then runs worker_fn and _exit()s -- this is
    // what prevents the fd-leak-across-siblings problem described above.
    for (size_t i = 0; i < shard_ids.size(); ++i) {
        pid_t pid = fork();
        if (pid < 0) {
            throw std::runtime_error("process_spawn: fork() failed");
        }
        if (pid == 0) {
            // Child: close all fds that aren't this shard's own worker_fd.
            for (size_t j = 0; j < pairs.size(); ++j) {
                if (j != i) {
                    ::close(pairs[j].router_fd);
                    ::close(pairs[j].worker_fd);
                } else {
                    ::close(pairs[j].router_fd);  // not ours to keep
                }
            }
            UnixSocketTransport transport(pairs[i].worker_fd);
            // Defensive safety net: worker_fn is a caller-supplied callback,
            // and this is INSIDE a forked child -- an uncaught exception
            // here must never be allowed to escape this scope and unwind
            // into whatever the PARENT process's call stack looked like at
            // fork() time. In a plain standalone process that would just
            // call std::terminate() and abort; found via testing that it's
            // worth being explicit and guaranteed here rather than relying
            // on that default, especially since _exit() below would
            // otherwise never be reached at all.
            int rc = 1;
            try {
                rc = worker_fn(transport, shard_ids[i]);
            } catch (...) {
                rc = 1;
            }
            _exit(rc);  // not exit() -- see file header
        }
        // Parent: record the child, keep going.
        result.child_pids.push_back(pid);
    }

    // Phase 3 (parent only): close every worker_fd (not ours), wrap every
    // router_fd in a transport the caller owns.
    for (size_t i = 0; i < shard_ids.size(); ++i) {
        ::close(pairs[i].worker_fd);
        result.transports[shard_ids[i]] =
            std::make_unique<UnixSocketTransport>(pairs[i].router_fd);
    }

    return result;
}

}  // namespace aether::orchestration
