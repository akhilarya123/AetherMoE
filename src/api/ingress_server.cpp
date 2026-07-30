// AetherMoE — src/api/ingress_server.cpp
#include "ingress_server.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <iostream>

namespace aether::api {

using json = nlohmann::json;

IngressServer::IngressServer(IngressQueue& queue,
                              SequenceStatusTable& status_table,
                              std::atomic<uint64_t>& next_seq_id)
    : queue_(queue), status_table_(status_table), next_seq_id_(next_seq_id) {}

void IngressServer::run(const std::string& host, int port) {
    httplib::Server server;

    server.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok", "text/plain");
    });

    server.Post("/generate", [this](const httplib::Request& req,
                                     httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"accepted", false},
                                  {"reason", std::string("invalid json: ") + e.what()}}
                                 .dump(),
                             "application/json");
            return;
        }

        if (!body.contains("prompt_tokens") || !body["prompt_tokens"].is_array()) {
            res.status = 400;
            res.set_content(
                json{{"accepted", false}, {"reason", "missing prompt_tokens array"}}
                    .dump(),
                "application/json");
            return;
        }

        std::vector<uint32_t> prompt_tokens;
        for (auto& t : body["prompt_tokens"]) {
            prompt_tokens.push_back(t.get<uint32_t>());
        }
        uint32_t max_new_tokens = body.value("max_new_tokens", 16);

        uint64_t seq_id = next_seq_id_.fetch_add(1, std::memory_order_relaxed);
        auto seq = std::make_shared<core::Sequence>(seq_id, std::move(prompt_tokens),
                                                       max_new_tokens);

        if (!queue_.try_push(seq)) {
            // Backpressure: the ring buffer is full, meaning the scheduler
            // can't keep up. Reject immediately (HTTP 503) rather than
            // blocking the ingress thread — the caller should retry with
            // backoff. This is the "no request-blocking under sustained
            // load" behavior required by the M1 DoD.
            res.status = 503;
            res.set_content(
                json{{"accepted", false}, {"reason", "ingress queue full"}}.dump(),
                "application/json");
            return;
        }

        status_table_.upsert(seq_id, core::SequencePhase::PREFILL, 0);
        res.set_content(json{{"accepted", true}, {"seq_id", seq_id}}.dump(),
                         "application/json");
    });

    server.Get(R"(/status/(\d+))", [this](const httplib::Request& req,
                                           httplib::Response& res) {
        uint64_t seq_id = std::stoull(req.matches[1]);
        SequenceStatusTable::Status status;
        if (!status_table_.get(seq_id, status)) {
            res.status = 404;
            res.set_content(json{{"error", "unknown seq_id"}}.dump(),
                             "application/json");
            return;
        }
        res.set_content(json{{"seq_id", seq_id},
                              {"phase", core::to_string(status.phase)},
                              {"generated_count", status.generated_count}}
                             .dump(),
                         "application/json");
    });

    std::cout << "[ingress] listening on " << host << ":" << port << std::endl;
    server.listen(host.c_str(), port);
}

void IngressServer::stop() { stop_requested_.store(true, std::memory_order_release); }

}  // namespace aether::api
