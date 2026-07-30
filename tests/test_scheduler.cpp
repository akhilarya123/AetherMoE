#include "core/scheduler.hpp"

#include <gtest/gtest.h>
#include <random>

using aether::core::ContinuousBatchingScheduler;
using aether::core::PagedKVCacheAllocator;
using aether::core::Sequence;
using aether::core::SequencePhase;
using aether::core::SchedulerConfig;

namespace {
std::vector<uint32_t> make_prompt(size_t n) {
    return std::vector<uint32_t>(n, 1);
}
}  // namespace

TEST(Scheduler, SinglePromptGoesPrefillThenDecodeThenFinished) {
    PagedKVCacheAllocator pt(16, 64);
    SchedulerConfig cfg;
    cfg.max_batch_tokens = 1024;
    cfg.max_prefill_chunk = 512;
    ContinuousBatchingScheduler sched(cfg, pt);

    auto seq = std::make_shared<Sequence>(1, make_prompt(10), /*max_new=*/3);
    sched.admit(seq);

    // Iteration 1: whole prompt (10 tokens) fits in one chunk -> DECODE.
    auto r1 = sched.step();
    ASSERT_EQ(r1.size(), 1u);
    EXPECT_TRUE(r1[0].was_prefill_chunk);
    EXPECT_EQ(r1[0].tokens_processed, 10u);
    EXPECT_EQ(seq->phase, SequencePhase::DECODE);

    // 3 decode iterations to reach max_new_tokens.
    for (int i = 0; i < 3; ++i) {
        auto r = sched.step();
        ASSERT_EQ(r.size(), 1u);
        EXPECT_FALSE(r[0].was_prefill_chunk);
    }
    EXPECT_EQ(seq->phase, SequencePhase::FINISHED);
    EXPECT_EQ(sched.num_active(), 0u);
}

TEST(Scheduler, LargePromptSpansMultiplePrefillChunks) {
    PagedKVCacheAllocator pt(64, 64);
    SchedulerConfig cfg;
    cfg.max_batch_tokens = 100;
    cfg.max_prefill_chunk = 40;  // forces 3 chunks for a 100-token prompt
    ContinuousBatchingScheduler sched(cfg, pt);

    auto seq = std::make_shared<Sequence>(1, make_prompt(100), /*max_new=*/1);
    sched.admit(seq);

    auto r1 = sched.step();
    EXPECT_EQ(r1[0].tokens_processed, 40u);
    EXPECT_EQ(seq->phase, SequencePhase::PREFILL);

    auto r2 = sched.step();
    EXPECT_EQ(r2[0].tokens_processed, 40u);
    EXPECT_EQ(seq->phase, SequencePhase::PREFILL);

    auto r3 = sched.step();
    EXPECT_EQ(r3[0].tokens_processed, 20u);
    EXPECT_EQ(seq->phase, SequencePhase::DECODE);
}

TEST(Scheduler, DecodeIsPrioritizedOverNewPrefillAdmission) {
    PagedKVCacheAllocator pt(64, 64);
    SchedulerConfig cfg;
    cfg.max_batch_tokens = 5;  // tiny budget forces prioritization to matter
    cfg.max_prefill_chunk = 5;
    ContinuousBatchingScheduler sched(cfg, pt);

    // seq A already decoding.
    auto a = std::make_shared<Sequence>(1, make_prompt(2), 5);
    sched.admit(a);
    sched.step();  // prefill A fully (2 tokens) -> DECODE
    ASSERT_EQ(a->phase, SequencePhase::DECODE);

    // seq B waiting to be admitted.
    auto b = std::make_shared<Sequence>(2, make_prompt(3), 1);
    sched.admit(b);

    auto r = sched.step();
    // A's decode step must appear in results (not starved), even though B is
    // waiting and budget is tight.
    bool a_decoded = false;
    for (auto& res : r) {
        if (res.seq_id == 1 && !res.was_prefill_chunk) a_decoded = true;
    }
    EXPECT_TRUE(a_decoded);
}

TEST(Scheduler, BlockedOnPageTableCapacityDoesNotCrashOrCorrupt) {
    // Only 1 physical block of 32 tokens total. A 16-token prompt + 1 decode
    // step = 17 tokens, which fits in that single block -- so the pool is
    // exactly enough for ONE sequence's full prefill+decode lifecycle, and
    // no more. A second sequence can't get capacity until the first
    // finishes and frees its block.
    PagedKVCacheAllocator pt(32, 1);
    SchedulerConfig cfg;
    cfg.max_batch_tokens = 1000;
    cfg.max_prefill_chunk = 1000;
    ContinuousBatchingScheduler sched(cfg, pt);

    auto a = std::make_shared<Sequence>(1, make_prompt(16), 1);
    auto b = std::make_shared<Sequence>(2, make_prompt(16), 1);
    sched.admit(a);
    sched.admit(b);

    auto r1 = sched.step();
    // Only seq A should have gotten capacity; B remains waiting.
    ASSERT_EQ(r1.size(), 1u);
    EXPECT_EQ(r1[0].seq_id, 1u);
    EXPECT_EQ(sched.num_waiting(), 1u);

    // Finish A's single decode step -> frees its block.
    auto r2 = sched.step();
    ASSERT_FALSE(r2.empty());
    EXPECT_EQ(a->phase, SequencePhase::FINISHED);

    // Now B should be able to get admitted.
    auto r3 = sched.step();
    ASSERT_EQ(r3.size(), 1u);
    EXPECT_EQ(r3[0].seq_id, 2u);
}

TEST(Scheduler, RandomizedWorkloadNeverThrowsAndAlwaysFinishesEventually) {
    PagedKVCacheAllocator pt(32, 256);
    SchedulerConfig cfg;
    cfg.max_batch_tokens = 128;
    cfg.max_prefill_chunk = 64;
    ContinuousBatchingScheduler sched(cfg, pt);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> prompt_len(1, 300);
    std::uniform_int_distribution<int> max_new(1, 20);

    for (uint64_t id = 0; id < 50; ++id) {
        sched.admit(std::make_shared<Sequence>(
            id, make_prompt(prompt_len(rng)), max_new(rng)));
    }

    int iterations = 0;
    while ((sched.num_active() > 0 || sched.num_waiting() > 0) &&
           iterations < 100000) {
        sched.step();
        ++iterations;
    }
    EXPECT_LT(iterations, 100000) << "scheduler appears to have stalled";
    EXPECT_EQ(sched.num_active(), 0u);
    EXPECT_EQ(sched.num_waiting(), 0u);
}
