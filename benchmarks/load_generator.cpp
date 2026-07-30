// AetherMoE — benchmarks/load_generator.cpp
//
// Milestone 1 Definition of Done requires demonstrating: 10,000+ req/s
// ingress throughput, 100+ concurrent in-flight requests, randomized
// 10-1000 token sequence lengths, with zero request-blocking (503s under
// backpressure are fine and expected — a *hang* is not) and no crashes
// over a sustained run.
//
// Usage:
//   ./load_generator <host> <port> <num_threads> <requests_per_thread>
//   ./load_generator 127.0.0.1 8080 100 200

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using json = nlohmann::json;

int main(int argc, char** argv) {
    std::string host = argc > 1 ? argv[1] : "127.0.0.1";
    int port = argc > 2 ? std::atoi(argv[2]) : 8080;
    int num_threads = argc > 3 ? std::atoi(argv[3]) : 100;
    int requests_per_thread = argc > 4 ? std::atoi(argv[4]) : 200;

    std::atomic<uint64_t> accepted{0};
    std::atomic<uint64_t> rejected{0};
    std::atomic<uint64_t> errored{0};

    std::cout << "[load_generator] " << num_threads << " threads x "
              << requests_per_thread << " requests each -> "
              << (num_threads * requests_per_thread) << " total requests, target "
              << host << ":" << port << "\n";

    auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t] {
            httplib::Client client(host, port);
            client.set_connection_timeout(5);
            client.set_read_timeout(5);

            std::mt19937 rng(t * 7919u + 12345u);
            std::uniform_int_distribution<int> len_dist(10, 1000);
            std::uniform_int_distribution<int> max_new_dist(1, 32);

            for (int i = 0; i < requests_per_thread; ++i) {
                int prompt_len = len_dist(rng);
                json body;
                body["prompt_tokens"] = std::vector<int>(prompt_len, 1);
                body["max_new_tokens"] = max_new_dist(rng);

                auto res = client.Post("/generate", body.dump(), "application/json");
                if (!res) {
                    errored.fetch_add(1, std::memory_order_relaxed);
                } else if (res->status == 200) {
                    accepted.fetch_add(1, std::memory_order_relaxed);
                } else if (res->status == 503) {
                    rejected.fetch_add(1, std::memory_order_relaxed);
                } else {
                    errored.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& w : workers) w.join();

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    uint64_t total = accepted + rejected + errored;

    std::cout << "\n--- results ---\n";
    std::cout << "total requests:   " << total << "\n";
    std::cout << "accepted (200):   " << accepted << "\n";
    std::cout << "backpressure(503):" << rejected << "\n";
    std::cout << "errors:           " << errored << "\n";
    std::cout << "wall time:        " << seconds << "s\n";
    std::cout << "throughput:       " << (total / seconds) << " req/s\n";
    return 0;
}
