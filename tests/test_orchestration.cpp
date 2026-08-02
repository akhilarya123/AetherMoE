#include "orchestration/process_spawn.hpp"
#include "orchestration/router.hpp"
#include "orchestration/worker_shard.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <random>
#include <thread>

using aether::orchestration::ICollectiveTransport;
using aether::orchestration::Router;
using aether::orchestration::RouterConfig;
using aether::orchestration::RoutedResult;
using aether::orchestration::RoutedToken;
using aether::orchestration::run_worker_loop;
using aether::orchestration::spawn_workers;
using aether::orchestration::SpawnedWorkers;

namespace {

// Uniform expert->shard mapping: experts split into contiguous, equal-sized
// ranges, one range per shard.
std::unordered_map<uint32_t, uint32_t> make_expert_to_shard(uint32_t num_experts,
                                                               uint32_t num_shards) {
    std::unordered_map<uint32_t, uint32_t> map;
    uint32_t experts_per_shard = num_experts / num_shards;
    for (uint32_t e = 0; e < num_experts; ++e) {
        map[e] = e / experts_per_shard;
    }
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

void expect_results_match(const std::vector<RoutedToken>& sent,
                            const std::vector<RoutedResult>& got,
                            const std::unordered_map<uint32_t, uint32_t>& expert_to_shard) {
    ASSERT_EQ(got.size(), sent.size());
    for (size_t i = 0; i < sent.size(); ++i) {
        const auto& t = sent[i];
        const auto& r = got[i];
        EXPECT_EQ(r.batch_position, i);
        EXPECT_EQ(r.token_id, t.token_id);
        // processed_by_shard must match the shard for effective_expert, NOT
        // necessarily primary_expert -- these differ exactly when a token
        // overflowed to its secondary_expert (Phase B). For any batch
        // routed with unlimited capacity, effective_expert == primary_expert
        // always, so this check is strictly more general, not a relaxation.
        EXPECT_EQ(r.processed_by_shard, expert_to_shard.at(r.effective_expert));
        ASSERT_EQ(r.payload.size(), t.payload.size());
        for (size_t k = 0; k < t.payload.size(); ++k) {
            EXPECT_FLOAT_EQ(r.payload[k], -t.payload[k]);
        }
    }
}

// Spawns num_shards real worker processes and returns a ready-to-use
// Router plus the SpawnedWorkers handle (caller must keep it alive for the
// router's transports to remain valid, and must call shut_down() before
// the handle goes out of scope).
struct TestCluster {
    SpawnedWorkers spawned;
    std::unordered_map<uint32_t, ICollectiveTransport*> shard_transports;
    std::unordered_map<uint32_t, uint32_t> expert_to_shard;
    std::unique_ptr<Router> router;

    void shut_down() {
        shard_transports.clear();
        spawned.transports.clear();  // closes fds -> workers see PeerClosedError
        auto statuses = spawned.join_all();
        for (int status : statuses) {
            EXPECT_TRUE(WIFEXITED(status));
            EXPECT_EQ(WEXITSTATUS(status), 0);
        }
    }
};

TestCluster make_cluster(uint32_t num_shards, uint32_t num_experts,
                           aether::orchestration::RouterConfig config = {}) {
    TestCluster cluster;
    cluster.expert_to_shard = make_expert_to_shard(num_experts, num_shards);

    std::vector<uint32_t> shard_ids;
    for (uint32_t s = 0; s < num_shards; ++s) shard_ids.push_back(s);

    cluster.spawned = spawn_workers(shard_ids,
        [](ICollectiveTransport& transport, uint32_t shard_id) -> int {
            run_worker_loop(transport, shard_id);
            return 0;
        });

    for (auto& [id, transport] : cluster.spawned.transports) {
        cluster.shard_transports[id] = transport.get();
    }
    cluster.router = std::make_unique<Router>(cluster.expert_to_shard, cluster.shard_transports, config);
    return cluster;
}

}  // namespace

TEST(Orchestration, EmptyBatchDoesNotHang) {
    auto cluster = make_cluster(/*num_shards=*/2, /*num_experts=*/4);
    auto results = cluster.router->route_batch({});
    EXPECT_TRUE(results.empty());
    cluster.shut_down();
}

TEST(Orchestration, SingleShardRoutesAllTokensCorrectly) {
    auto cluster = make_cluster(/*num_shards=*/1, /*num_experts=*/4);
    std::mt19937 rng(1);
    uint64_t next_id = 1;
    auto batch = make_random_batch(200, 4, rng, next_id);
    auto results = cluster.router->route_batch(batch);
    expect_results_match(batch, results, cluster.expert_to_shard);
    cluster.shut_down();
}

TEST(Orchestration, MultiShardScattersAndReassemblesInOrder) {
    auto cluster = make_cluster(/*num_shards=*/4, /*num_experts=*/16);
    std::mt19937 rng(2);
    uint64_t next_id = 1;
    auto batch = make_random_batch(500, 16, rng, next_id);
    auto results = cluster.router->route_batch(batch);
    expect_results_match(batch, results, cluster.expert_to_shard);
    cluster.shut_down();
}

TEST(Orchestration, MultipleBatchesThroughSameWorkers) {
    auto cluster = make_cluster(/*num_shards=*/3, /*num_experts=*/6);
    std::mt19937 rng(3);
    uint64_t next_id = 1;
    for (int b = 0; b < 10; ++b) {
        auto batch = make_random_batch(50 + b * 7, 6, rng, next_id);
        auto results = cluster.router->route_batch(batch);
        expect_results_match(batch, results, cluster.expert_to_shard);
    }
    cluster.shut_down();
}

// Per the M3 testing plan: synthetic traffic where a large fraction of
// tokens select the same expert. Phase A has no capacity-ceiling overflow
// yet (that's Phase B) -- this confirms scatter/gather/reassembly still
// works correctly under heavy skew, not that load is balanced.
TEST(Orchestration, SkewedDistributionStillReassemblesCorrectly) {
    auto cluster = make_cluster(/*num_shards=*/4, /*num_experts=*/16);
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> skew_roll(0.0f, 1.0f);
    uint64_t next_id = 1;

    std::vector<RoutedToken> batch;
    size_t n = 400;
    for (size_t i = 0; i < n; ++i) {
        uint32_t expert = (skew_roll(rng) < 0.8f) ? 0 : 1;  // 80%+ to expert 0
        batch.emplace_back(next_id++, static_cast<uint32_t>(i), expert, expert,
                             std::vector<float>{val_dist(rng)});
    }
    auto results = cluster.router->route_batch(batch);
    expect_results_match(batch, results, cluster.expert_to_shard);
    cluster.shut_down();
}

// ===================== Phase B: expert-capacity ceiling =====================

namespace {

// Independently recomputes expected effective_expert per token using the
// SAME rule Router applies (first `capacity` per primary_expert, ranked by
// batch_position, win; the rest overflow to their own secondary_expert) --
// deliberately reimplemented here rather than calling into Router's
// private logic, so this test can catch a real bug in that logic rather
// than just echoing it back.
std::vector<uint32_t> expected_effective_experts(const std::vector<RoutedToken>& tokens,
                                                   size_t capacity) {
    std::vector<uint32_t> effective(tokens.size());
    std::unordered_map<uint32_t, std::vector<size_t>> by_primary;
    for (size_t i = 0; i < tokens.size(); ++i) {
        by_primary[tokens[i].primary_expert].push_back(i);
    }
    for (auto& [expert, indices] : by_primary) {
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return tokens[a].batch_position < tokens[b].batch_position;
        });
        for (size_t rank = 0; rank < indices.size(); ++rank) {
            size_t idx = indices[rank];
            effective[idx] = (rank < capacity) ? tokens[idx].primary_expert
                                                 : tokens[idx].secondary_expert;
        }
    }
    return effective;
}

}  // namespace

TEST(Orchestration, CapacityCeilingDefaultIsUnlimited_NoBehaviorChange) {
    // Default RouterConfig{} must reproduce exact Phase A behavior --
    // capacity ceiling is opt-in, not a silent change for existing callers.
    auto cluster = make_cluster(/*num_shards=*/4, /*num_experts=*/16);  // default config
    std::mt19937 rng(10);
    uint64_t next_id = 1;
    auto batch = make_random_batch(500, 16, rng, next_id);
    auto results = cluster.router->route_batch(batch);
    for (size_t i = 0; i < batch.size(); ++i) {
        // unlimited capacity must never reroute to secondary_expert
        EXPECT_EQ(results[i].effective_expert, batch[i].primary_expert);
    }
    expect_results_match(batch, results, cluster.expert_to_shard);
    cluster.shut_down();
}

TEST(Orchestration, OverflowReroutesToSecondaryExpert_ExactAssignmentMatch) {
    RouterConfig config;
    config.expert_capacity = 20;
    auto cluster = make_cluster(/*num_shards=*/4, /*num_experts=*/16, config);

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<uint32_t> secondary_dist(1, 15);  // never 0 -- see below
    uint64_t next_id = 1;

    // 400 of 500 tokens choose expert 0 as primary (>>20, guarantees heavy
    // overflow); each gets a distinct, valid secondary in [1, 15] so
    // overflow has somewhere sensible to land.
    std::vector<RoutedToken> batch;
    size_t n = 500;
    for (size_t i = 0; i < n; ++i) {
        bool wants_zero = i < 400;
        uint32_t primary = wants_zero ? 0 : (1 + static_cast<uint32_t>(i) % 15);
        uint32_t secondary = secondary_dist(rng);
        batch.emplace_back(next_id++, static_cast<uint32_t>(i), primary, secondary,
                             std::vector<float>{val_dist(rng)});
    }

    auto results = cluster.router->route_batch(batch);
    auto expected = expected_effective_experts(batch, config.expert_capacity);

    ASSERT_EQ(results.size(), batch.size());
    size_t overflowed_count = 0;
    for (size_t i = 0; i < batch.size(); ++i) {
        SCOPED_TRACE("token index " + std::to_string(i));
        EXPECT_EQ(results[i].effective_expert, expected[i]);
        EXPECT_EQ(results[i].processed_by_shard,
                   cluster.expert_to_shard.at(expected[i]));
        if (expected[i] != batch[i].primary_expert) {
            ++overflowed_count;
            EXPECT_EQ(expected[i], batch[i].secondary_expert);
        }
    }
    // Expert 0 got 400 requests against a capacity of 20 -> exactly 380
    // should have overflowed to their secondary.
    EXPECT_EQ(overflowed_count, 400u - config.expert_capacity);

    expect_results_match(batch, results, cluster.expert_to_shard);
    cluster.shut_down();
}

// Per the M3 testing plan: confirm the capacity ceiling reroutes overflow
// correctly AND that no node stalls the pipeline -- a generous wall-clock
// bound catches an actual hang (e.g. a shard waiting on a shard nobody
// sent it anything, or a reroute landing on an unregistered shard) rather
// than assuming "it returned" is enough.
TEST(Orchestration, LoadImbalanceStressTest_NoStall) {
    RouterConfig config;
    config.expert_capacity = 15;
    auto cluster = make_cluster(/*num_shards=*/8, /*num_experts=*/32, config);

    std::mt19937 rng(12);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> hot_roll(0.0f, 1.0f);
    std::uniform_int_distribution<uint32_t> secondary_dist(1, 31);
    uint64_t next_id = 1;

    auto start = std::chrono::steady_clock::now();

    for (int round = 0; round < 8; ++round) {
        std::vector<RoutedToken> batch;
        size_t n = 800;
        for (size_t i = 0; i < n; ++i) {
            // 90% pile onto expert 0 -- deliberately far beyond any
            // reasonable capacity, to stress overflow handling hard.
            uint32_t primary = (hot_roll(rng) < 0.9f) ? 0 : (1 + static_cast<uint32_t>(i) % 31);
            uint32_t secondary = secondary_dist(rng);
            batch.emplace_back(next_id++, static_cast<uint32_t>(i), primary, secondary,
                                 std::vector<float>{val_dist(rng)});
        }
        auto results = cluster.router->route_batch(batch);
        expect_results_match(batch, results, cluster.expert_to_shard);
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    // 8 rounds of heavily-skewed 800-token batches should not take anywhere
    // near this long -- a much larger number here would indicate a stall.
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 10);

    cluster.shut_down();
}
// for by the spec's concurrency/deadlock testing plan, not just a single
// happy-path shape.
// ===================== Fault injection (detection only) =====================
//
// Per the spec: fault injection is a TESTING PLAN item for Milestone 3,
// confirming the router DETECTS a dead node rather than hanging -- full
// graceful degradation/recovery is explicitly Milestone 4's job, not
// implemented or tested here. This validates existing behavior (the
// transport already throws on a closed/dead peer -- see
// UnixSocketTransport::read_all), not new production code.

TEST(Orchestration, KilledWorkerIsDetectedNotHung) {
    auto cluster = make_cluster(/*num_shards=*/4, /*num_experts=*/16);
    std::mt19937 rng(20);
    uint64_t next_id = 1;

    // Prove the cluster is healthy first -- a fault detected because the
    // cluster was never really working wouldn't tell us anything.
    auto healthy_batch = make_random_batch(200, 16, rng, next_id);
    auto healthy_results = cluster.router->route_batch(healthy_batch);
    expect_results_match(healthy_batch, healthy_results, cluster.expert_to_shard);

    // shard_ids were 0..num_shards-1 in that order when passed to
    // spawn_workers (see make_cluster), so child_pids[1] is shard 1's pid.
    ASSERT_EQ(cluster.spawned.child_pids.size(), 4u);
    pid_t victim_pid = cluster.spawned.child_pids[1];
    ::kill(victim_pid, SIGKILL);
    // Give the kernel a moment to actually tear the process down and close
    // its socket fd before we route more work at it.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Route a batch that includes tokens for shard 1 (the dead one) among
    // others -- must detect the failure (throw) within a bounded time, not
    // hang waiting on a receive() that will never come.
    auto batch2 = make_random_batch(200, 16, rng, next_id);
    bool threw = false;
    auto start = std::chrono::steady_clock::now();
    try {
        cluster.router->route_batch(batch2);
    } catch (const std::exception&) {
        threw = true;
    }
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_TRUE(threw);
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 5);

    // Custom shutdown (not cluster.shut_down()): the killed child exited via
    // SIGKILL, not a clean return 0, so it must be checked differently from
    // the other three, which should still have exited cleanly once their
    // transports closed.
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

TEST(Orchestration, RandomizedShapesRepeatedRuns) {
    struct Config { uint32_t shards; uint32_t experts; };
    std::vector<Config> configs = {
        {1, 1}, {2, 4}, {4, 16}, {3, 3}, {8, 16}, {4, 4},
    };
    std::mt19937 rng(12345);

    for (auto& cfg : configs) {
        for (int rep = 0; rep < 5; ++rep) {
            SCOPED_TRACE("shards=" + std::to_string(cfg.shards) +
                          " experts=" + std::to_string(cfg.experts) +
                          " rep=" + std::to_string(rep));
            auto cluster = make_cluster(cfg.shards, cfg.experts);
            uint64_t next_id = 1;
            std::uniform_int_distribution<size_t> size_dist(0, 500);
            for (int b = 0; b < 5; ++b) {
                size_t n = size_dist(rng);
                auto batch = make_random_batch(n, cfg.experts, rng, next_id);
                auto results = cluster.router->route_batch(batch);
                expect_results_match(batch, results, cluster.expert_to_shard);
            }
            cluster.shut_down();
        }
    }
}
