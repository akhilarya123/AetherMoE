// AetherMoE — src/core/spmc_ring_buffer.hpp
//
// Lock-free ring buffer used as the ingress distribution channel: the
// control-plane thread (single producer) pushes incoming requests; a pool of
// worker/scheduler threads (multiple consumers) pop them.
//
// Algorithm: Dmitry Vyukov's bounded MPMC queue. It is correct for any
// producer/consumer count, so it trivially covers the SPMC case we need here
// while leaving room to add more ingress producers later without a rewrite.
//
// Design notes:
//   - Capacity must be a power of two (enforced with static_assert) so index
//     wraparound is a cheap mask instead of a modulo.
//   - Each cell carries its own "sequence" counter. A push/pop only succeeds
//     when the cell's sequence matches the expected turn, which is what
//     makes this safe under multiple concurrent consumers without a lock.
//   - Head/tail cursors are cache-line padded (alignas(64)) so the producer
//     cursor and consumer cursor don't false-share a cache line, which would
//     otherwise silently serialize independent cores through cache coherence
//     traffic.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace aether::core {

template <typename T, std::size_t Capacity>
class LockFreeRingBuffer {
    static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two >= 2");
    static_assert(std::is_nothrow_move_constructible_v<T> ||
                      std::is_copy_constructible_v<T>,
                  "T must be movable or copyable");

public:
    LockFreeRingBuffer() {
        for (std::size_t i = 0; i < Capacity; ++i) {
            cells_[i].sequence.store(i, std::memory_order_relaxed);
        }
        enqueue_pos_.store(0, std::memory_order_relaxed);
        dequeue_pos_.store(0, std::memory_order_relaxed);
    }

    LockFreeRingBuffer(const LockFreeRingBuffer&) = delete;
    LockFreeRingBuffer& operator=(const LockFreeRingBuffer&) = delete;

    // Returns false if the buffer is full (backpressure signal to caller).
    bool try_push(T item) {
        Cell* cell;
        std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &cells_[pos & kMask];
            std::size_t seq = cell->sequence.load(std::memory_order_acquire);
            std::intptr_t diff =
                static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);
            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;  // full
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        cell->data = std::move(item);
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    // Returns false if the buffer is empty.
    bool try_pop(T& out) {
        Cell* cell;
        std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &cells_[pos & kMask];
            std::size_t seq = cell->sequence.load(std::memory_order_acquire);
            std::intptr_t diff = static_cast<std::intptr_t>(seq) -
                                  static_cast<std::intptr_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;  // empty
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
        out = std::move(cell->data);
        cell->sequence.store(pos + kMask + 1, std::memory_order_release);
        return true;
    }

    static constexpr std::size_t capacity() { return Capacity; }

    // Approximate size — for metrics/telemetry only, NOT for correctness
    // decisions (it can be stale the instant it's read under concurrency).
    std::size_t size_approx() const {
        std::size_t enq = enqueue_pos_.load(std::memory_order_relaxed);
        std::size_t deq = dequeue_pos_.load(std::memory_order_relaxed);
        return enq >= deq ? enq - deq : 0;
    }

private:
    static constexpr std::size_t kMask = Capacity - 1;

    struct Cell {
        std::atomic<std::size_t> sequence;
        T data;
    };

    // alignas(64) prevents false sharing between the producer-owned
    // enqueue_pos_ and the consumer-owned dequeue_pos_: without this padding
    // both cursors could land in the same cache line, and every push would
    // invalidate the cache line every consumer just read, and vice versa.
    alignas(64) std::atomic<std::size_t> enqueue_pos_;
    alignas(64) std::atomic<std::size_t> dequeue_pos_;
    alignas(64) Cell cells_[Capacity];
};

}  // namespace aether::core
