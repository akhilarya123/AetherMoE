// AetherMoE — src/core/page_table.hpp
//
// A from-scratch, minimal version of PagedAttention's memory model: the
// KV-cache is carved into fixed-size physical "blocks" (each holding
// BLOCK_SIZE tokens' worth of key/value slots). Sequences are allocated
// blocks lazily, one at a time, as their token count grows past the current
// block boundary — instead of reserving a worst-case contiguous buffer up
// front. This is what eliminates the classic fragmentation problem of naive
// KV-cache allocation: a 10-token request and a 900-token request no longer
// need to be pre-sized into equally large slabs.
//
// Concurrency: guarded by a single mutex. Block allocation happens far less
// often than ring-buffer push/pop (once per BLOCK_SIZE tokens per sequence,
// not once per token), so a mutex is the right first cut — a lock-free
// free-list (Treiber stack) is a documented stretch goal, not required for
// Milestone 1 correctness.
//
// "Zero fragmentation" here means: no physical block is ever reserved for a
// sequence that isn't using at least one slot in it, and no block is held by
// more than one sequence. Internal fragmentation (a block's last few slots
// unused) is bounded by BLOCK_SIZE - 1 tokens per sequence, which is the
// same bound PagedAttention itself accepts.

#pragma once

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace aether::core {

class PagedKVCacheAllocator {
public:
    PagedKVCacheAllocator(size_t block_size_tokens, size_t num_physical_blocks)
        : block_size_(block_size_tokens), num_blocks_(num_physical_blocks) {
        if (block_size_ == 0 || num_blocks_ == 0) {
            throw std::invalid_argument(
                "block_size and num_physical_blocks must be > 0");
        }
        free_blocks_.reserve(num_blocks_);
        // Push in descending order so blocks are handed out 0,1,2,... which
        // makes test assertions and debugging easier (not required for
        // correctness).
        for (size_t i = num_blocks_; i-- > 0;) {
            free_blocks_.push_back(static_cast<int>(i));
        }
    }

    size_t block_size() const { return block_size_; }
    size_t num_physical_blocks() const { return num_blocks_; }

    size_t free_block_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        return free_blocks_.size();
    }

    // How many blocks `token_count` tokens need, given block_size_.
    size_t blocks_needed_for(size_t token_count) const {
        return (token_count + block_size_ - 1) / block_size_;
    }

    // Grows a sequence's allocation to cover `token_count` tokens, allocating
    // additional physical blocks as needed. Returns false (and allocates
    // nothing) if there isn't enough free capacity — the caller (scheduler)
    // is expected to treat this as "cannot admit more work this iteration",
    // not a fatal error.
    bool ensure_capacity(uint64_t seq_id, size_t token_count) {
        std::lock_guard<std::mutex> lock(mu_);
        auto& owned = tables_[seq_id];  // creates empty entry if new
        size_t needed = blocks_needed_for(token_count);
        if (needed <= owned.size()) return true;  // already enough

        size_t to_allocate = needed - owned.size();
        if (free_blocks_.size() < to_allocate) {
            return false;  // OOM: not enough physical blocks right now
        }
        for (size_t i = 0; i < to_allocate; ++i) {
            owned.push_back(free_blocks_.back());
            free_blocks_.pop_back();
        }
        return true;
    }

    // Returns all blocks owned by seq_id to the free list. Safe to call on
    // an unknown seq_id (no-op) so callers don't need a separate "did this
    // sequence ever allocate" check.
    void free_sequence(uint64_t seq_id) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = tables_.find(seq_id);
        if (it == tables_.end()) return;
        for (int block : it->second) {
            free_blocks_.push_back(block);
        }
        tables_.erase(it);
    }

    std::vector<int> blocks_of(uint64_t seq_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = tables_.find(seq_id);
        return it == tables_.end() ? std::vector<int>{} : it->second;
    }

    // Fraction of allocated capacity that's unused padding within the last
    // block of each active sequence, as a sanity metric for the load test
    // ("0% fragmentation" claim in the Definition of Done — this is what
    // that assertion actually measures against).
    double internal_fragmentation_ratio(
        const std::unordered_map<uint64_t, size_t>& seq_token_counts) const {
        std::lock_guard<std::mutex> lock(mu_);
        size_t allocated_slots = 0;
        size_t used_slots = 0;
        for (const auto& [seq_id, owned] : tables_) {
            allocated_slots += owned.size() * block_size_;
            auto it = seq_token_counts.find(seq_id);
            used_slots += (it == seq_token_counts.end()) ? 0 : it->second;
        }
        if (allocated_slots == 0) return 0.0;
        return 1.0 - (static_cast<double>(used_slots) / allocated_slots);
    }

private:
    size_t block_size_;
    size_t num_blocks_;
    mutable std::mutex mu_;
    std::vector<int> free_blocks_;                       // stack of free physical block ids
    std::unordered_map<uint64_t, std::vector<int>> tables_;  // seq_id -> owned physical blocks
};

}  // namespace aether::core
