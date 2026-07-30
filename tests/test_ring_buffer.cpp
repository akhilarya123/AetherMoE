#include "core/spmc_ring_buffer.hpp"

#include <atomic>
#include <gtest/gtest.h>
#include <numeric>
#include <thread>
#include <unordered_set>
#include <vector>

using aether::core::LockFreeRingBuffer;

TEST(RingBuffer, PushPopSingleThreadedFIFO) {
    LockFreeRingBuffer<int, 8> rb;
    for (int i = 0; i < 8; ++i) {
        ASSERT_TRUE(rb.try_push(i));
    }
    // Full now.
    ASSERT_FALSE(rb.try_push(999));

    for (int i = 0; i < 8; ++i) {
        int out = -1;
        ASSERT_TRUE(rb.try_pop(out));
        EXPECT_EQ(out, i);
    }
    int out;
    ASSERT_FALSE(rb.try_pop(out));  // empty now
}

TEST(RingBuffer, WrapAroundReuse) {
    LockFreeRingBuffer<int, 4> rb;
    for (int round = 0; round < 100; ++round) {
        ASSERT_TRUE(rb.try_push(round));
        int out;
        ASSERT_TRUE(rb.try_pop(out));
        EXPECT_EQ(out, round);
    }
}

TEST(RingBuffer, SizeApproxTracksPushesAndPops) {
    LockFreeRingBuffer<int, 16> rb;
    EXPECT_EQ(rb.size_approx(), 0u);
    rb.try_push(1);
    rb.try_push(2);
    EXPECT_EQ(rb.size_approx(), 2u);
    int out;
    rb.try_pop(out);
    EXPECT_EQ(rb.size_approx(), 1u);
}

// Single-producer / multi-consumer stress test: one producer pushes N
// sequential ids; several consumer threads race to pop them. Correctness
// requires that every id 0..N-1 is popped exactly once, with no drops and no
// duplicates, regardless of thread interleaving.
TEST(RingBuffer, SPMCStressNoDropsNoDuplicates) {
    constexpr size_t kCapacity = 1024;
    constexpr int kTotal = 200000;
    constexpr int kConsumers = 8;

    LockFreeRingBuffer<int, kCapacity> rb;
    std::atomic<bool> producer_done{false};
    std::vector<std::vector<int>> consumed(kConsumers);

    std::thread producer([&] {
        for (int i = 0; i < kTotal; ++i) {
            while (!rb.try_push(i)) {
                std::this_thread::yield();  // backpressure: spin until space frees
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    std::vector<std::thread> consumers;
    for (int c = 0; c < kConsumers; ++c) {
        consumers.emplace_back([&, c] {
            int val;
            for (;;) {
                if (rb.try_pop(val)) {
                    consumed[c].push_back(val);
                } else if (producer_done.load(std::memory_order_acquire)) {
                    // Producer finished; drain any remaining items then exit.
                    if (!rb.try_pop(val)) break;
                    consumed[c].push_back(val);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    producer.join();
    for (auto& t : consumers) t.join();

    std::unordered_set<int> seen;
    size_t total_consumed = 0;
    for (auto& v : consumed) {
        for (int x : v) {
            auto [it, inserted] = seen.insert(x);
            ASSERT_TRUE(inserted) << "duplicate value popped: " << x;
        }
        total_consumed += v.size();
    }
    EXPECT_EQ(total_consumed, static_cast<size_t>(kTotal));
    EXPECT_EQ(seen.size(), static_cast<size_t>(kTotal));
}

TEST(RingBuffer, CapacityReportedCorrectly) {
    LockFreeRingBuffer<int, 32> rb;
    EXPECT_EQ(rb.capacity(), 32u);
}
