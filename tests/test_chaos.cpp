#include "orchestration/chaos.hpp"
#include "orchestration/process_spawn.hpp"
#include "orchestration/router.hpp"
#include "orchestration/worker_shard.hpp"

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <thread>

using aether::orchestration::ChaosScript;
using aether::orchestration::DelayInjectingTransport;
using aether::orchestration::ICollectiveTransport;
using aether::orchestration::Router;
using aether::orchestration::RoutedResult;
using aether::orchestration::RoutedToken;
using aether::orchestration::run_worker_loop;
using aether::orchestration::spawn_workers;
using aether::orchestration::SpawnedWorkers;

namespace {

// Same helpers as test_orchestration.cpp, duplicated rather than shared
// across translation units -- these are ~15-line test fixtures, not
// production code, and a shared test-utility header would be more
// machinery than the duplication it avoids.
std::unordered_map<uint32_t, uint32_t> make_expert_to_shard(uint32_t num_experts, uint32_t num_shards) {
    std::unordered_map<uint32_t, uint32_t> map;
    uint32_t experts_per_shard = num_experts / num_shards;
    for (uint32_t e = 0; e < num_experts; ++e) map[e] = e / experts_per_shard;
    return map;
}

std::vector<RoutedToken> make_random_batch(size_t n, uint32_t num_experts,
                                             std::mt19937& rng, uint64_t& next_token_id) {
    std::uniform_int_distribution<uint32_t> expert_dist(0, num_experts - 1);
    std::uniform_real_distribution<float> val_dist(-10.0f, 10.0f);
    std::vector<RoutedToken> batch;
    batch.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        std::vector<float> payload = {val_dist(rng), val_dist(rng), val_dist(rng)};
        batch.emplace_back(next_token_id++, static_cast<uint32_t>(i),
                            expert_dist(rng), expert_dist(rng), std::move(payload));
    }
    return batch;
}

struct TestCluster {
    SpawnedWorkers spawned;
    std::unordered_map<uint32_t, ICollectiveTransport*> shard_transports;
    std::unordered_map<uint32_t, uint32_t> expert_to_shard;
    std::unique_ptr<Router> router;

    void shut_down() {
        shard_transports.clear();
        spawned.transports.clear();
        auto statuses = spawned.join_all();
        for (int status : statuses) {
            EXPECT_TRUE(WIFEXITED(status));
            EXPECT_EQ(WEXITSTATUS(status), 0);
        }
    }
};

TestCluster make_cluster(uint32_t num_shards, uint32_t num_experts) {
    TestCluster cluster;
    cluster.expert_to_shard = make_expert_to_shard(num_experts, num_shards);
    std::vector<uint32_t> shard_ids;
    for (uint32_t s = 0; s < num_shards; ++s) shard_ids.push_back(s);
    cluster.spawned = spawn_workers(shard_ids, [](ICollectiveTransport& transport, uint32_t shard_id) -> int {
        run_worker_loop(transport, shard_id);
        return 0;
    });
    for (auto& [id, transport] : cluster.spawned.transports) {
        cluster.shard_transports[id] = transport.get();
    }
    cluster.router = std::make_unique<Router>(cluster.expert_to_shard, cluster.shard_transports);
    return cluster;
}

}  // namespace

// ===================== process_spawn.hpp: shard_pids =====================

TEST(Chaos, ShardPidsMapMatchesChildPidsByShardId) {
    auto cluster = make_cluster(4, 16);
    ASSERT_EQ(cluster.spawned.shard_pids.size(), 4u);
    // shard_ids were 0..3 in that order when passed to spawn_workers (see
    // make_cluster), so child_pids[s] must equal shard_pids[s] for every s.
    for (uint32_t s = 0; s < 4; ++s) {
        ASSERT_TRUE(cluster.spawned.shard_pids.count(s));
        EXPECT_EQ(cluster.spawned.shard_pids.at(s), cluster.spawned.child_pids[s]);
    }
    cluster.shut_down();
}

// ===================== chaos.hpp: DelayInjectingTransport =====================

TEST(Chaos, DelayInjectingTransportAddsMeasurableLatency) {
    auto cluster = make_cluster(1, 4);
    DelayInjectingTransport delayed(*cluster.shard_transports[0], std::chrono::milliseconds(80),
                                     std::chrono::milliseconds(80));
    std::unordered_map<uint32_t, ICollectiveTransport*> wrapped = {{0, &delayed}};
    Router router(cluster.expert_to_shard, wrapped);

    std::mt19937 rng(1);
    uint64_t next_id = 1;
    auto batch = make_random_batch(10, 4, rng, next_id);

    auto start = std::chrono::steady_clock::now();
    auto results = router.route_batch(batch);
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_EQ(results.size(), 10u);
    // 80ms send delay + 80ms receive delay, with slack for scheduling
    // jitter -- not a tight bound, just enough to prove the delay is real
    // and roughly the right order of magnitude, not that it's exact.
    EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 150);
    cluster.shut_down();
}

TEST(Chaos, DelayInjectingTransportZeroDelayIsPureNoOp) {
    // Default-constructed delays (0ms) must behave identically to talking
    // to the wrapped transport directly -- proves the decorator doesn't
    // change correctness, only latency.
    auto cluster = make_cluster(1, 4);
    DelayInjectingTransport wrapped_transport(*cluster.shard_transports[0]);
    std::unordered_map<uint32_t, ICollectiveTransport*> wrapped = {{0, &wrapped_transport}};
    Router router(cluster.expert_to_shard, wrapped);

    std::mt19937 rng(5);
    uint64_t next_id = 1;
    auto batch = make_random_batch(50, 4, rng, next_id);
    auto results = router.route_batch(batch);
    ASSERT_EQ(results.size(), batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        EXPECT_EQ(results[i].token_id, batch[i].token_id);
        EXPECT_EQ(results[i].batch_position, i);
    }
    cluster.shut_down();
}

// ===================== router.hpp: route_batch_tolerant =====================

TEST(Chaos, TolerantRoutingMatchesStrictRoutingWhenClusterIsHealthy) {
    auto cluster = make_cluster(4, 16);
    std::mt19937 rng(2);
    uint64_t next_id = 1;
    auto batch = make_random_batch(300, 16, rng, next_id);
    auto results = cluster.router->route_batch_tolerant(batch);

    ASSERT_EQ(results.size(), batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        EXPECT_FALSE(results[i].failed);
        EXPECT_TRUE(results[i].failure_reason.empty());
        EXPECT_EQ(results[i].token_id, batch[i].token_id);
        EXPECT_EQ(results[i].batch_position, i);
    }
    EXPECT_EQ(cluster.router->unhealthy_shard_count(), 0u);
    cluster.shut_down();
}

TEST(Chaos, TolerantRoutingIsolatesADeadShardWithoutThrowingOrHanging) {
    auto cluster = make_cluster(4, 16);
    std::mt19937 rng(4);
    uint64_t next_id = 1;

    // Prove healthy first, same discipline as
    // Orchestration.KilledWorkerIsDetectedNotHung.
    auto healthy_batch = make_random_batch(200, 16, rng, next_id);
    auto healthy_results = cluster.router->route_batch_tolerant(healthy_batch);
    for (auto& r : healthy_results) EXPECT_FALSE(r.failed);

    pid_t victim_pid = cluster.spawned.child_pids[1];
    ::kill(victim_pid, SIGKILL);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto batch2 = make_random_batch(200, 16, rng, next_id);
    auto start = std::chrono::steady_clock::now();
    std::vector<RoutedResult> results;
    ASSERT_NO_THROW(results = cluster.router->route_batch_tolerant(batch2));
    auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 5);

    ASSERT_EQ(results.size(), batch2.size());
    size_t failed_count = 0, ok_count = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        // Every position accounted for, whatever the outcome -- this is
        // the property that actually matters, more than the specific
        // failed/succeeded split.
        EXPECT_EQ(results[i].batch_position, i);
        EXPECT_EQ(results[i].token_id, batch2[i].token_id);
        if (results[i].failed) {
            ++failed_count;
            EXPECT_FALSE(results[i].failure_reason.empty());
        } else {
            ++ok_count;
        }
    }
    EXPECT_GT(failed_count, 0u) << "expected at least some tokens routed to the dead shard";
    EXPECT_GT(ok_count, 0u) << "expected the three surviving shards to keep serving";
    EXPECT_TRUE(cluster.router->is_shard_unhealthy(1));

    // A THIRD batch, now that shard 1 is already known-unhealthy, must
    // resolve fast (no actual attempt to talk to the dead socket) --
    // this is the "isolated" half of "isolate the fault," not just
    // "detect it once."
    auto batch3 = make_random_batch(200, 16, rng, next_id);
    auto start3 = std::chrono::steady_clock::now();
    std::vector<RoutedResult> results3;
    ASSERT_NO_THROW(results3 = cluster.router->route_batch_tolerant(batch3));
    auto elapsed3 = std::chrono::steady_clock::now() - start3;
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed3).count(), 1000);
    EXPECT_EQ(results3.size(), batch3.size());

    cluster.shard_transports.clear();
    cluster.spawned.transports.clear();
    auto statuses = cluster.spawned.join_all();
    ASSERT_EQ(statuses.size(), 4u);
    EXPECT_TRUE(WIFSIGNALED(statuses[1]));
    EXPECT_EQ(WTERMSIG(statuses[1]), SIGKILL);
    for (size_t i = 0; i < statuses.size(); ++i) {
        if (i == 1) continue;
        EXPECT_TRUE(WIFEXITED(statuses[i]));
        EXPECT_EQ(WEXITSTATUS(statuses[i]), 0);
    }
}

// ===================== chaos.hpp: ChaosScript during active traffic =====================
//
// The M4 spec specifically asks for kills "at random points during active
// traffic," not just before a test starts (Milestone 3's fault-injection
// test already covers the latter). This drives many batches back-to-back
// on the main thread while a background thread kills a shard at a random
// point partway through -- proving the whole run never hangs, never
// crashes the test process, and accounts for every token sent across the
// entire run, not just the one batch that happens to straddle the kill.
TEST(Chaos, ScriptedKillDuringActiveTrafficNeverHangsAndAccountsForEveryToken) {
    constexpr uint32_t kNumShards = 4;
    constexpr uint32_t kNumExperts = 16;
    constexpr uint32_t kVictimShard = 1;
    auto cluster = make_cluster(kNumShards, kNumExperts);

    ChaosScript chaos;
    std::mt19937 chaos_rng(99);
    chaos.schedule_kill(cluster.spawned, kVictimShard, std::chrono::milliseconds(150), chaos_rng);

    std::mt19937 rng(3);
    uint64_t next_id = 1;
    uint64_t total_sent = 0, total_succeeded = 0, total_failed = 0;
    bool saw_victim_marked_unhealthy = false;

    for (int batch_num = 0; batch_num < 40; ++batch_num) {
        auto batch = make_random_batch(200, kNumExperts, rng, next_id);
        total_sent += batch.size();

        auto batch_start = std::chrono::steady_clock::now();
        std::vector<RoutedResult> results;
        ASSERT_NO_THROW(results = cluster.router->route_batch_tolerant(batch));
        auto batch_elapsed = std::chrono::steady_clock::now() - batch_start;
        ASSERT_LT(std::chrono::duration_cast<std::chrono::seconds>(batch_elapsed).count(), 5);

        ASSERT_EQ(results.size(), batch.size());
        for (auto& r : results) {
            if (r.failed) ++total_failed;
            else ++total_succeeded;
        }
        if (cluster.router->is_shard_unhealthy(kVictimShard)) saw_victim_marked_unhealthy = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_EQ(total_succeeded + total_failed, total_sent)
        << "every token sent must resolve to exactly one outcome -- none missing, none double-counted";
    EXPECT_TRUE(saw_victim_marked_unhealthy);
    EXPECT_TRUE(cluster.router->is_shard_unhealthy(kVictimShard));
    EXPECT_GT(total_failed, 0u);
    EXPECT_GT(total_succeeded, 0u)
        << "the three surviving shards must keep serving traffic throughout, not just before the kill";

    chaos.join_all();
    cluster.shard_transports.clear();
    cluster.spawned.transports.clear();
    auto statuses = cluster.spawned.join_all();
    ASSERT_EQ(statuses.size(), kNumShards);
    EXPECT_TRUE(WIFSIGNALED(statuses[kVictimShard]));
    EXPECT_EQ(WTERMSIG(statuses[kVictimShard]), SIGKILL);
    for (size_t i = 0; i < statuses.size(); ++i) {
        if (i == kVictimShard) continue;
        EXPECT_TRUE(WIFEXITED(statuses[i]));
        EXPECT_EQ(WEXITSTATUS(statuses[i]), 0);
    }
}
