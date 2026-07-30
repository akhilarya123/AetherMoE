#include "core/page_table.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using aether::core::PagedKVCacheAllocator;

TEST(PageTable, AllocatesBlocksLazilyAsSequenceGrows) {
    PagedKVCacheAllocator pt(/*block_size_tokens=*/16, /*num_physical_blocks=*/4);
    EXPECT_EQ(pt.free_block_count(), 4u);

    // 10 tokens fits in 1 block of size 16.
    ASSERT_TRUE(pt.ensure_capacity(/*seq_id=*/1, 10));
    EXPECT_EQ(pt.blocks_of(1).size(), 1u);
    EXPECT_EQ(pt.free_block_count(), 3u);

    // Growing to 20 tokens needs a 2nd block.
    ASSERT_TRUE(pt.ensure_capacity(1, 20));
    EXPECT_EQ(pt.blocks_of(1).size(), 2u);
    EXPECT_EQ(pt.free_block_count(), 2u);

    // Shrinking the *requested* size below current capacity should not
    // release blocks (ensure_capacity only grows).
    ASSERT_TRUE(pt.ensure_capacity(1, 5));
    EXPECT_EQ(pt.blocks_of(1).size(), 2u);
}

TEST(PageTable, FreeSequenceReturnsBlocksToPool) {
    PagedKVCacheAllocator pt(16, 4);
    ASSERT_TRUE(pt.ensure_capacity(1, 32));  // 2 blocks
    EXPECT_EQ(pt.free_block_count(), 2u);

    pt.free_sequence(1);
    EXPECT_EQ(pt.free_block_count(), 4u);
    EXPECT_TRUE(pt.blocks_of(1).empty());
}

TEST(PageTable, FreeUnknownSequenceIsNoOp) {
    PagedKVCacheAllocator pt(16, 4);
    EXPECT_NO_THROW(pt.free_sequence(999));
    EXPECT_EQ(pt.free_block_count(), 4u);
}

TEST(PageTable, ExhaustionReturnsFalseWithoutPartialAllocation) {
    PagedKVCacheAllocator pt(16, 2);  // only 32 tokens total capacity
    ASSERT_TRUE(pt.ensure_capacity(1, 16));   // 1 block used
    ASSERT_TRUE(pt.ensure_capacity(2, 16));   // 2nd block used, pool exhausted

    // seq 3 needs 2 blocks (17-32 tokens), but 0 are free.
    EXPECT_FALSE(pt.ensure_capacity(3, 17));
    // No partial allocation should have happened for seq 3.
    EXPECT_TRUE(pt.blocks_of(3).empty());
    EXPECT_EQ(pt.free_block_count(), 0u);
}

TEST(PageTable, NoBlockSharedBetweenTwoSequences) {
    PagedKVCacheAllocator pt(8, 16);
    for (uint64_t seq = 0; seq < 8; ++seq) {
        ASSERT_TRUE(pt.ensure_capacity(seq, 8));  // 1 block each = 8 blocks total
    }
    std::vector<int> all_blocks;
    for (uint64_t seq = 0; seq < 8; ++seq) {
        for (int b : pt.blocks_of(seq)) all_blocks.push_back(b);
    }
    std::sort(all_blocks.begin(), all_blocks.end());
    for (size_t i = 1; i < all_blocks.size(); ++i) {
        EXPECT_NE(all_blocks[i], all_blocks[i - 1]) << "block reused across sequences";
    }
}

TEST(PageTable, FragmentationRatioComputation) {
    PagedKVCacheAllocator pt(16, 4);
    // seq 1: uses 10 of 16 slots in its single block -> 6/16 wasted
    ASSERT_TRUE(pt.ensure_capacity(1, 10));
    std::unordered_map<uint64_t, size_t> usage{{1, 10}};
    double frag = pt.internal_fragmentation_ratio(usage);
    EXPECT_NEAR(frag, 6.0 / 16.0, 1e-9);
}

TEST(PageTable, BlocksNeededForRounding) {
    PagedKVCacheAllocator pt(16, 100);
    EXPECT_EQ(pt.blocks_needed_for(0), 0u);
    EXPECT_EQ(pt.blocks_needed_for(1), 1u);
    EXPECT_EQ(pt.blocks_needed_for(16), 1u);
    EXPECT_EQ(pt.blocks_needed_for(17), 2u);
    EXPECT_EQ(pt.blocks_needed_for(32), 2u);
}

// Concurrency smoke test: many threads growing/freeing distinct sequences
// simultaneously should never corrupt the free list (no negative counts, no
// double-allocation) — run under ThreadSanitizer in CI (see CMakeLists).
TEST(PageTable, ConcurrentGrowAndFreeIsRaceFree) {
    PagedKVCacheAllocator pt(16, 64);
    constexpr int kThreads = 16;
    constexpr int kItersPerThread = 200;

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kItersPerThread; ++i) {
                uint64_t seq_id = static_cast<uint64_t>(t) * 1000000 + i;
                if (pt.ensure_capacity(seq_id, 16)) {
                    pt.free_sequence(seq_id);
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(pt.free_block_count(), 64u);
}
