// AetherMoE — src/orchestration/serialization.hpp
//
// Manual binary (de)serialization for batches of RoutedToken/RoutedResult.
// No protobuf/flatbuffers/json dependency -- deliberate, not just a sandbox
// convenience: a real collective-communication library moves raw framed
// bytes, not JSON, so hand-rolled binary framing is actually the more
// faithful simulation here, not a shortcut.
//
// HONESTY NOTE: payload floats/counts are packed in host-native byte order
// (plain memcpy), not an explicit endianness. This is safe ONLY because
// every "node" in this milestone is a process on the SAME machine/
// architecture by construction -- the simulation's entire premise. A real
// multi-machine transport would need explicit endianness handling here the
// way the outer transport framing already does (see
// UnixSocketTransport::encode_length) for exactly that reason. Flagging
// this rather than silently relying on it.

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "routed_token.hpp"

namespace aether::orchestration {

namespace detail {

inline void put_u32(std::vector<uint8_t>& buf, uint32_t v) {
    size_t off = buf.size();
    buf.resize(off + 4);
    std::memcpy(buf.data() + off, &v, 4);
}
inline void put_u64(std::vector<uint8_t>& buf, uint64_t v) {
    size_t off = buf.size();
    buf.resize(off + 8);
    std::memcpy(buf.data() + off, &v, 8);
}
inline void put_floats(std::vector<uint8_t>& buf, const std::vector<float>& v) {
    put_u32(buf, static_cast<uint32_t>(v.size()));
    if (!v.empty()) {
        size_t off = buf.size();
        buf.resize(off + v.size() * sizeof(float));
        std::memcpy(buf.data() + off, v.data(), v.size() * sizeof(float));
    }
}

struct Reader {
    const uint8_t* p;
    const uint8_t* end;

    void need(size_t n) const {
        if (static_cast<size_t>(end - p) < n) {
            throw std::runtime_error("serialization: truncated message (framing bug)");
        }
    }
    uint32_t get_u32() {
        need(4);
        uint32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    uint64_t get_u64() {
        need(8);
        uint64_t v;
        std::memcpy(&v, p, 8);
        p += 8;
        return v;
    }
    std::vector<float> get_floats() {
        uint32_t n = get_u32();
        need(static_cast<size_t>(n) * sizeof(float));
        std::vector<float> v(n);
        if (n > 0) std::memcpy(v.data(), p, n * sizeof(float));
        p += static_cast<size_t>(n) * sizeof(float);
        return v;
    }
};

}  // namespace detail

inline std::vector<uint8_t> encode_token_batch(const std::vector<RoutedToken>& tokens) {
    std::vector<uint8_t> buf;
    detail::put_u32(buf, static_cast<uint32_t>(tokens.size()));
    for (const auto& t : tokens) {
        detail::put_u64(buf, t.token_id);
        detail::put_u32(buf, t.batch_position);
        detail::put_u32(buf, t.primary_expert);
        detail::put_u32(buf, t.secondary_expert);
        detail::put_floats(buf, t.payload);
    }
    return buf;
}

inline std::vector<RoutedToken> decode_token_batch(const std::vector<uint8_t>& bytes) {
    detail::Reader r{bytes.data(), bytes.data() + bytes.size()};
    uint32_t count = r.get_u32();
    std::vector<RoutedToken> out;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        RoutedToken t;
        t.token_id = r.get_u64();
        t.batch_position = r.get_u32();
        t.primary_expert = r.get_u32();
        t.secondary_expert = r.get_u32();
        t.payload = r.get_floats();
        out.push_back(std::move(t));
    }
    return out;
}

inline std::vector<uint8_t> encode_result_batch(const std::vector<RoutedResult>& results) {
    std::vector<uint8_t> buf;
    detail::put_u32(buf, static_cast<uint32_t>(results.size()));
    for (const auto& r : results) {
        detail::put_u64(buf, r.token_id);
        detail::put_u32(buf, r.batch_position);
        detail::put_u32(buf, r.processed_by_shard);
        detail::put_floats(buf, r.payload);
    }
    return buf;
}

inline std::vector<RoutedResult> decode_result_batch(const std::vector<uint8_t>& bytes) {
    detail::Reader r{bytes.data(), bytes.data() + bytes.size()};
    uint32_t count = r.get_u32();
    std::vector<RoutedResult> out;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        RoutedResult res;
        res.token_id = r.get_u64();
        res.batch_position = r.get_u32();
        res.processed_by_shard = r.get_u32();
        res.payload = r.get_floats();
        out.push_back(std::move(res));
    }
    return out;
}

}  // namespace aether::orchestration
