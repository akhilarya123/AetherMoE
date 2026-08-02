cmake_minimum_required(VERSION 3.20)
project(AetherMoE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Sanitizer support: build with -DAETHER_SANITIZE=thread or =address to run
# the concurrency correctness suite under TSan/ASan, per the Milestone 1
# testing plan.
#   cmake -B build -DAETHER_SANITIZE=thread && cmake --build build
# ---------------------------------------------------------------------------
set(AETHER_SANITIZE "" CACHE STRING "Sanitizer to build with: thread, address, or empty")
if(AETHER_SANITIZE STREQUAL "thread")
    add_compile_options(-fsanitize=thread -g -O1)
    add_link_options(-fsanitize=thread)
elseif(AETHER_SANITIZE STREQUAL "address")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g -O1)
    add_link_options(-fsanitize=address)
endif()

if(NOT CMAKE_BUILD_TYPE AND AETHER_SANITIZE STREQUAL "")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra)

include(FetchContent)

# ---------------------------------------------------------------------------
# GoogleTest (unit + concurrency tests)
# ---------------------------------------------------------------------------
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---------------------------------------------------------------------------
# cpp-httplib (header-only HTTP server) for the ingress API — zero-cost,
# MIT-licensed, no gRPC/protobuf toolchain required for Milestone 1.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.5
)
FetchContent_MakeAvailable(httplib)

# ---------------------------------------------------------------------------
# nlohmann/json (header-only) for request/response bodies on the ingress API.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# ---------------------------------------------------------------------------
# Core library (header-only for now — everything lives in src/core/*.hpp)
# ---------------------------------------------------------------------------
add_library(aether_core INTERFACE)
target_include_directories(aether_core INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(aether_core INTERFACE Threads::Threads)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Engine binary: ingress API + ring buffer + scheduler wired together.
# ---------------------------------------------------------------------------
add_executable(aether_engine
    src/api/ingress_server.cpp
    src/main.cpp
)
target_link_libraries(aether_engine PRIVATE aether_core httplib::httplib nlohmann_json::nlohmann_json Threads::Threads)
# cpp-httplib's default TCP listen backlog is 5 (CPPHTTPLIB_LISTEN_BACKLOG),
# which silently drops (ECONNREFUSED) connections beyond that queue depth —
# fatal for the M1 load test's 100+ concurrent client requirement. Raise it
# well above our target concurrency.
target_compile_definitions(aether_engine PRIVATE CPPHTTPLIB_LISTEN_BACKLOG=1024)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_executable(aether_tests
    tests/test_ring_buffer.cpp
    tests/test_page_table.cpp
    tests/test_scheduler.cpp
    tests/test_orchestration.cpp
)
target_link_libraries(aether_tests PRIVATE aether_core GTest::gtest_main Threads::Threads)
target_include_directories(aether_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(GoogleTest)
gtest_discover_tests(aether_tests)

# ---------------------------------------------------------------------------
# Load generator (used by the Milestone 1 load test: 10,000+ req/s,
# randomized 10-1000 token sequence lengths, 100+ concurrent requests)
# ---------------------------------------------------------------------------
add_executable(load_generator benchmarks/load_generator.cpp)
target_link_libraries(load_generator PRIVATE aether_core httplib::httplib nlohmann_json::nlohmann_json Threads::Threads)
