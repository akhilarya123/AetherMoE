// AetherMoE — src/orchestration/unix_socket_transport.hpp
//
// ICollectiveTransport implementation over an already-connected AF_UNIX
// socket (SOCK_STREAM). The router creates connected socket PAIRS via
// socketpair() before forking each worker -- each side inherits its own fd
// across fork() and closes the unused end. This sidesteps named-socket
// filesystem paths, cleanup, and connect-before-bind race conditions
// entirely, which matters a lot for the spec's "repeated runs under
// randomized process-start orderings" concurrency test: there is no
// ordering to race, the fds already exist and are already connected before
// either process's real work begins.
//
// SOCK_STREAM has no message boundaries on its own -- two send() calls can
// arrive as one read(), or one send() can arrive as several reads. Framing
// with an explicit length prefix (big-endian uint64, chosen over relying on
// host endianness matching, since a real transport would need this too) is
// what turns a byte stream back into discrete messages.

#pragma once

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

#include "collective_transport.hpp"

namespace aether::orchestration {

class UnixSocketTransport : public ICollectiveTransport {
public:
    explicit UnixSocketTransport(int fd) : fd_(fd) {
        // BUG FOUND VIA REAL FAULT-INJECTION TESTING, not anticipated in
        // advance: writing to a socket whose peer has died (e.g. SIGKILLed)
        // raises SIGPIPE, whose default disposition TERMINATES THE PROCESS
        // -- before write_all() below ever gets a chance to see EPIPE and
        // throw a catchable exception. The fault-injection test's process
        // died with exit code 141 (128+SIGPIPE) instead of the exception
        // its own try/catch was supposed to observe.
        //
        // Deliberately NOT using MSG_NOSIGNAL (the tempting one-line Linux
        // fix): it doesn't exist on macOS, which is this entire project's
        // actual target platform, and this sandbox has no way to verify
        // anything Mac-specific directly. signal(SIGPIPE, SIG_IGN) is
        // standard POSIX and behaves identically on both -- the safe
        // choice specifically because of that asymmetry, not just the
        // simpler one. Idempotent, so calling it from every constructor
        // instance is harmless and doesn't need a call-once guard.
        std::signal(SIGPIPE, SIG_IGN);
    }

    UnixSocketTransport(const UnixSocketTransport&) = delete;
    UnixSocketTransport& operator=(const UnixSocketTransport&) = delete;

    ~UnixSocketTransport() override {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    void send(const std::vector<uint8_t>& message) override {
        uint8_t header[8];
        encode_length(message.size(), header);
        write_all(header, sizeof(header));
        if (!message.empty()) {
            write_all(message.data(), message.size());
        }
    }

    std::vector<uint8_t> receive() override {
        uint8_t header[8];
        read_all(header, sizeof(header), /*allow_eof_at_start=*/true);
        uint64_t len = decode_length(header);

        std::vector<uint8_t> message(len);
        if (len > 0) {
            read_all(message.data(), len, /*allow_eof_at_start=*/false);
        }
        return message;
    }

    int fd() const { return fd_; }

private:
    static void encode_length(size_t len, uint8_t out[8]) {
        uint64_t v = static_cast<uint64_t>(len);
        for (int i = 7; i >= 0; --i) {
            out[i] = static_cast<uint8_t>(v & 0xFF);
            v >>= 8;
        }
    }

    static uint64_t decode_length(const uint8_t in[8]) {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v = (v << 8) | in[i];
        }
        return v;
    }

    // Writes exactly `len` bytes, looping through partial writes and
    // retrying on EINTR. Throws on any other error or on the peer having
    // gone away (EPIPE).
    void write_all(const void* data, size_t len) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        size_t written = 0;
        while (written < len) {
            ssize_t n = ::write(fd_, p + written, len - written);
            if (n > 0) {
                written += static_cast<size_t>(n);
                continue;
            }
            if (n < 0 && errno == EINTR) continue;
            throw std::runtime_error(
                std::string("UnixSocketTransport::send failed: ") + std::strerror(errno));
        }
    }

    // Reads exactly `len` bytes, looping through partial reads and retrying
    // on EINTR. `allow_eof_at_start` distinguishes "peer closed cleanly
    // before sending anything" (expected, throws PeerClosedError so callers
    // like Router::gather can treat it as "this shard is done") from "peer
    // closed mid-message" (a real protocol/framing bug, throws
    // runtime_error instead -- should never happen if both sides agree on
    // framing, and silently truncating a partial message would be far worse
    // than failing loudly).
    void read_all(void* data, size_t len, bool allow_eof_at_start) {
        uint8_t* p = static_cast<uint8_t*>(data);
        size_t got = 0;
        while (got < len) {
            ssize_t n = ::read(fd_, p + got, len - got);
            if (n > 0) {
                got += static_cast<size_t>(n);
                continue;
            }
            if (n == 0) {
                if (allow_eof_at_start && got == 0) {
                    throw PeerClosedError();
                }
                throw std::runtime_error(
                    "UnixSocketTransport::receive: peer closed mid-message "
                    "(framing bug, not a clean close)");
            }
            if (errno == EINTR) continue;
            throw std::runtime_error(
                std::string("UnixSocketTransport::receive failed: ") + std::strerror(errno));
        }
    }

    int fd_;
};

}  // namespace aether::orchestration
