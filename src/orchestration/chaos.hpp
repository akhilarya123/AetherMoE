// AetherMoE — src/orchestration/chaos.hpp
//
// Milestone 4, Step 2: chaos-testing primitives for the orchestration
// layer. Two independent tools, matching the M4 spec's testing-plan item
// ("scripted process kills at random points during active traffic" +
// "introduces artificial IPC delays"):
//
//   1. DelayInjectingTransport -- a decorator over ICollectiveTransport
//      that sleeps for a configurable duration before send()/receive().
//      This reuses collective_transport.hpp's own "wrap it behind a small
//      interface" design exactly as intended: Router only ever holds an
//      ICollectiveTransport*, so substituting a delay-injecting wrapper
//      for one shard's entry needs zero changes to Router or
//      UnixSocketTransport.
//
//   2. ChaosScript -- schedules a real SIGKILL of a real worker process at
//      a random point within a time window, on a background thread, while
//      the caller's main thread keeps driving traffic. This is Milestone
//      3's own KilledWorkerIsDetectedNotHung test (kill-before-a-batch)
//      generalized to kill-DURING-active-traffic, which is what "at random
//      points during active traffic" actually requires -- a fixed kill
//      timed before the test even starts traffic doesn't exercise the
//      in-flight-request case at all.
//
// What this file deliberately does NOT do: decide how a caller should
// react to a chaos event. That's Router::route_batch_tolerant's job (see
// router.hpp) and the caller's own retry/backoff policy, if any -- this
// file only injects the fault, it doesn't interpret it, same separation of
// concerns as ICollectiveTransport itself not knowing anything about
// capacity ceilings or expert routing.

#pragma once

#include <signal.h>

#include <chrono>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "collective_transport.hpp"
#include "process_spawn.hpp"

namespace aether::orchestration {

// Wraps an existing ICollectiveTransport& (non-owning -- this class never
// manages the wrapped transport's lifetime, matching how Router itself
// only ever borrows transports via raw pointers) and adds a fixed,
// artificial delay before send()/receive(). A send_delay simulates a slow
// outbound link; a receive_delay simulates a slow/backed-up peer. Both
// default to zero (no-op passthrough) so a caller can inject delay on only
// one side of one shard's traffic without needing two separate wrapper
// types.
class DelayInjectingTransport : public ICollectiveTransport {
public:
    DelayInjectingTransport(ICollectiveTransport& inner,
                             std::chrono::milliseconds send_delay = std::chrono::milliseconds(0),
                             std::chrono::milliseconds receive_delay = std::chrono::milliseconds(0))
        : inner_(inner), send_delay_(send_delay), receive_delay_(receive_delay) {}

    void send(const std::vector<uint8_t>& message) override {
        if (send_delay_.count() > 0) std::this_thread::sleep_for(send_delay_);
        inner_.send(message);
    }

    std::vector<uint8_t> receive() override {
        if (receive_delay_.count() > 0) std::this_thread::sleep_for(receive_delay_);
        return inner_.receive();
    }

    // Chaos scripts adjust delay live (e.g. "healthy for the first second,
    // then degrade") without needing to reconstruct the wrapper -- and
    // Router holds transports by pointer, so live mutation here is visible
    // to the router immediately, no re-registration needed.
    void set_send_delay(std::chrono::milliseconds d) { send_delay_ = d; }
    void set_receive_delay(std::chrono::milliseconds d) { receive_delay_ = d; }

private:
    ICollectiveTransport& inner_;
    std::chrono::milliseconds send_delay_;
    std::chrono::milliseconds receive_delay_;
};

// Schedules real SIGKILLs of real worker processes at random points within
// a time window, without blocking the caller's own thread -- the caller
// keeps driving traffic on its own thread while kills happen concurrently
// in the background, which is what makes this "during active traffic"
// rather than "before the test starts."
//
// Every scheduled kill runs on its own std::thread; the destructor joins
// all of them, so a ChaosScript going out of scope always leaves no
// dangling background work behind, the same discipline
// TelemetryFlusher's destructor uses for its own background thread.
class ChaosScript {
public:
    ChaosScript() = default;
    ~ChaosScript() { join_all(); }

    ChaosScript(const ChaosScript&) = delete;
    ChaosScript& operator=(const ChaosScript&) = delete;

    // Kills `shard_id`'s worker process after a delay drawn uniformly from
    // [0, window]. Looks up the pid via SpawnedWorkers::shard_pids (see
    // process_spawn.hpp) rather than requiring the caller to track pids
    // itself.
    void schedule_kill(SpawnedWorkers& workers, uint32_t shard_id,
                        std::chrono::milliseconds window, std::mt19937& rng) {
        auto it = workers.shard_pids.find(shard_id);
        if (it == workers.shard_pids.end()) {
            throw std::runtime_error(
                "ChaosScript: no known pid for shard_id " + std::to_string(shard_id));
        }
        pid_t pid = it->second;
        std::chrono::milliseconds delay =
            (window.count() <= 0)
                ? std::chrono::milliseconds(0)
                : std::chrono::milliseconds(
                      std::uniform_int_distribution<long long>(0, window.count())(rng));
        schedule_kill_after(pid, delay);
    }

    // Lower-level form: kill an already-known pid after a fixed delay.
    // Exposed directly (not just via schedule_kill) so a test can assert
    // exact timing without fighting a random window.
    void schedule_kill_after(pid_t pid, std::chrono::milliseconds delay) {
        threads_.emplace_back([pid, delay] {
            if (delay.count() > 0) std::this_thread::sleep_for(delay);
            ::kill(pid, SIGKILL);
        });
    }

    void join_all() {
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
        threads_.clear();
    }

private:
    std::vector<std::thread> threads_;
};

}  // namespace aether::orchestration
