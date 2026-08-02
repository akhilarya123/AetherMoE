// AetherMoE — tools/double_buffer_demo.mm
//
// Milestone 3, Phase C: double-buffered execution using Metal command
// queues/command buffers and MTLSharedEvent, with GPU-side event signaling
// instead of CPU-side blocking waits between slices -- per the spec:
// "while one token slice is being processed by the GPU, the next slice's
// (simulated) transfer proceeds concurrently."
//
// ============================== HONESTY NOTE ==============================
// I cannot run, compile, or test Metal/Objective-C++ code in my sandbox --
// no Apple Silicon, no macOS, no Metal framework. Every line below is
// written from documented Metal API knowledge, not verified execution.
// This is a materially bigger leap of faith than Milestone 2's Python/MLX
// kernels (at least Python is something that exists broadly) or Milestone
// 3 Phases A/B (which I compiled and ran myself, extensively, with real
// bugs found and fixed along the way). Treat this the way you'd treat a
// first draft from a careful engineer who has never had a compiler in the
// loop: plausible, reasoned through deliberately, but genuinely capable of
// being wrong in ways only your Mac can reveal. Please paste back the
// EXACT compiler error or runtime output if anything goes wrong, the same
// discipline that made Milestone 2's bugs fixable in one shot.
// ===========================================================================
//
// DESIGN
// ------
// Two independent MTLCommandQueues -- one for "transfer" (a blit copy,
// standing in for a host->device transfer), one for "compute" (a trivial
// placeholder kernel, NOT real MoE math -- that's Milestone 2's job,
// integrated later). Independent queues, rather than one queue with
// interleaved command buffers, because that's what actually gives the GPU
// driver room to schedule them concurrently -- command buffers on the SAME
// queue are not guaranteed to overlap even without explicit dependencies.
//
// Classic double-buffering: TWO device-local (MTLResourceStorageModePrivate)
// input buffers, alternating by slice index (slice i uses buffer i%2). All
// NUM_SLICES worth of input data is prepared into ONE shared staging buffer
// UP FRONT, before the pipeline starts -- this sidesteps an entire class of
// staging-buffer synchronization complexity (no need to double-buffer the
// staging area itself, since it's never overwritten mid-pipeline; realistic
// too, since a real inference batch's tokens are already host-resident
// before the pipeline begins).
//
// Synchronization is ONE MTLSharedEvent with a monotonically increasing
// counter, entirely GPU-side:
//   transferDoneValue(i) = 2*i + 1   -- transfer(i) signals this
//   computeDoneValue(i)  = 2*i + 2   -- compute(i)  signals this
//
// Per slice i:
//   transfer(i): if i >= 2, GPU-side wait for computeDoneValue(i-2) (don't
//                overwrite a buffer compute(i-2) might still be reading),
//                then blit-copy staging[i] -> deviceInput[i%2],
//                then signal transferDoneValue(i).
//   compute(i):  GPU-side wait for transferDoneValue(i), then run the
//                kernel on deviceInput[i%2] -> output[i], then signal
//                computeDoneValue(i).
//
// This is what creates the actual overlap: compute(i) can be running on
// the compute queue at the SAME time transfer(i+1) is running on the
// transfer queue, since transfer(i+1) only depends on computeDoneValue(i-1)
// (already satisfied), not computeDoneValue(i). No CPU code is involved in
// that coordination at all -- encodeWaitForEvent/encodeSignalEvent are
// GPU-side commands. The ONLY CPU-side blocking wait in this entire file is
// a single waitUntilCompleted on the LAST command buffer, once, at the very
// end, purely so the program knows when to read back results and exit --
// not part of the per-slice pipeline coordination the spec is asking about.
//
// CORRECTNESS: kernel is `out = in*in + 1.0f`, deliberately simple and
// synthetic (matches the placeholder-compute convention already established
// in worker_shard.hpp's process_token_placeholder -- Milestone 3 proves the
// PIPELINE mechanics, not real expert math). Results are read back and
// checked against a CPU-computed reference after the final sync, so a
// silent correctness bug can't hide behind "it ran without crashing".
//
// TUNING: ELEMENTS_PER_SLICE controls how much work each transfer/compute
// step does. If Instruments' Metal System Trace shows windows too short to
// read clearly, increase this constant and rerun -- it's deliberately a
// single tunable, not something to rederive.
//
// BUILD (no CMake integration yet, per the "standalone tool" decision):
//   clang++ -std=c++17 -fobjc-arc -framework Metal -framework Foundation \
//       tools/double_buffer_demo.mm -o double_buffer_demo
//   ./double_buffer_demo
//
// INSTRUMENTS EVIDENCE (the actual Definition-of-Done requirement):
//   Profile ./double_buffer_demo with the Metal System Trace template
//   (same procedure as Milestone 2's Instruments walkthrough). Look at the
//   GPU timeline for the transfer queue's blit intervals and the compute
//   queue's compute intervals -- overlap means seeing a compute(i)
//   interval's time window genuinely intersect a transfer(i+1) interval's
//   window, not one starting only after the other ends.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static const int NUM_SLICES = 8;
// Default 1M floats/slice (4MB); override via argv[1] (element count) to test
// whether larger per-slice work gives the driver more room to overlap
// transfer and compute across the two queues -- see chat for why.
static size_t ELEMENTS_PER_SLICE = 1u << 20;

static const char* kKernelSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void placeholder_expert_compute(device const float* in [[buffer(0)]],
                                        device float* out [[buffer(1)]],
                                        uint id [[thread_position_in_grid]]) {
    float x = in[id];
    out[id] = x * x + 1.0f;
}
)";

int main(int argc, char** argv) {
    if (argc > 1) {
        long override = std::atol(argv[1]);
        if (override > 0) {
            ELEMENTS_PER_SLICE = static_cast<size_t>(override);
        } else {
            std::fprintf(stderr, "Ignoring invalid argv[1]='%s', using default.\n", argv[1]);
        }
    }
    const size_t SLICE_BYTES = ELEMENTS_PER_SLICE * sizeof(float);

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::fprintf(stderr, "FATAL: MTLCreateSystemDefaultDevice() returned nil "
                                  "-- no Metal-capable GPU found.\n");
            return 1;
        }
        std::fprintf(stderr, "Device: %s\n", device.name.UTF8String);

        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@(kKernelSource)
                                                        options:nil
                                                          error:&error];
        if (!library) {
            std::fprintf(stderr, "FATAL: newLibraryWithSource failed: %s\n",
                          error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"placeholder_expert_compute"];
        if (!function) {
            std::fprintf(stderr, "FATAL: could not find function placeholder_expert_compute "
                                  "in compiled library.\n");
            return 1;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            std::fprintf(stderr, "FATAL: newComputePipelineStateWithFunction failed: %s\n",
                          error.localizedDescription.UTF8String);
            return 1;
        }

        id<MTLCommandQueue> transferQueue = [device newCommandQueue];
        id<MTLCommandQueue> computeQueue = [device newCommandQueue];
        if (!transferQueue || !computeQueue) {
            std::fprintf(stderr, "FATAL: newCommandQueue failed.\n");
            return 1;
        }

        id<MTLSharedEvent> event = [device newSharedEvent];
        if (!event) {
            std::fprintf(stderr, "FATAL: newSharedEvent failed.\n");
            return 1;
        }

        // Staging: ALL slices' input data, prepared up front (see design
        // note above for why this avoids double-buffering the staging area).
        id<MTLBuffer> staging = [device newBufferWithLength:(NUM_SLICES * SLICE_BYTES)
                                                      options:MTLResourceStorageModeShared];
        // Output: one region per slice, CPU-readable directly (kernel
        // writes here, no blit-back needed).
        id<MTLBuffer> output = [device newBufferWithLength:(NUM_SLICES * SLICE_BYTES)
                                                     options:MTLResourceStorageModeShared];
        // Double-buffered device-local inputs -- the actual "double buffer"
        // this phase is named for.
        id<MTLBuffer> deviceInput[2];
        deviceInput[0] = [device newBufferWithLength:SLICE_BYTES
                                              options:MTLResourceStorageModePrivate];
        deviceInput[1] = [device newBufferWithLength:SLICE_BYTES
                                              options:MTLResourceStorageModePrivate];
        if (!staging || !output || !deviceInput[0] || !deviceInput[1]) {
            std::fprintf(stderr, "FATAL: buffer allocation failed.\n");
            return 1;
        }

        // Fill staging with synthetic input data and keep a CPU-side mirror
        // for the correctness check after readback.
        std::vector<float> cpu_reference(NUM_SLICES * ELEMENTS_PER_SLICE);
        {
            float* staging_ptr = static_cast<float*>(staging.contents);
            for (int i = 0; i < NUM_SLICES; ++i) {
                for (size_t j = 0; j < ELEMENTS_PER_SLICE; ++j) {
                    float v = static_cast<float>((i * 1000003 + j) % 997) * 0.01f;
                    staging_ptr[i * ELEMENTS_PER_SLICE + j] = v;
                    cpu_reference[i * ELEMENTS_PER_SLICE + j] = v * v + 1.0f;
                }
            }
        }

        std::fprintf(stderr, "Encoding %d slices (%.1f MB each) across two command queues...\n",
                      NUM_SLICES, SLICE_BYTES / (1024.0 * 1024.0));

        id<MTLCommandBuffer> lastComputeCB = nil;

        for (int i = 0; i < NUM_SLICES; ++i) {
            int idx = i % 2;
            uint64_t transferDoneValue = 2ull * i + 1;
            uint64_t computeDoneValue = 2ull * i + 2;

            // ---- Transfer command buffer ----
            id<MTLCommandBuffer> transferCB = [transferQueue commandBuffer];
            transferCB.label = [NSString stringWithFormat:@"transfer[%d]", i];

            if (i >= 2) {
                // Don't overwrite deviceInput[idx] until compute(i-2), the
                // last consumer of this physical buffer, has finished
                // reading it. GPU-side wait, not a CPU stall.
                uint64_t mustWaitFor = 2ull * (i - 2) + 2;
                [transferCB encodeWaitForEvent:event value:mustWaitFor];
            }

            id<MTLBlitCommandEncoder> blit = [transferCB blitCommandEncoder];
            [blit copyFromBuffer:staging
                     sourceOffset:(i * SLICE_BYTES)
                         toBuffer:deviceInput[idx]
                destinationOffset:0
                             size:SLICE_BYTES];
            [blit endEncoding];

            [transferCB encodeSignalEvent:event value:transferDoneValue];
            [transferCB commit];

            // ---- Compute command buffer ----
            id<MTLCommandBuffer> computeCB = [computeQueue commandBuffer];
            computeCB.label = [NSString stringWithFormat:@"compute[%d]", i];

            [computeCB encodeWaitForEvent:event value:transferDoneValue];

            id<MTLComputeCommandEncoder> enc = [computeCB computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:deviceInput[idx] offset:0 atIndex:0];
            [enc setBuffer:output offset:(i * SLICE_BYTES) atIndex:1];

            NSUInteger maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
            NSUInteger tgSize = std::min<NSUInteger>(256, maxThreadsPerGroup);
            MTLSize gridSize = MTLSizeMake(ELEMENTS_PER_SLICE, 1, 1);
            MTLSize threadgroupSize = MTLSizeMake(tgSize, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [enc endEncoding];

            [computeCB encodeSignalEvent:event value:computeDoneValue];
            [computeCB commit];

            if (i == NUM_SLICES - 1) {
                lastComputeCB = computeCB;
            }
        }

        std::fprintf(stderr, "All slices submitted. Waiting for the final compute "
                              "command buffer only (one CPU-side wait, at the very "
                              "end -- not part of the per-slice pipeline)...\n");
        [lastComputeCB waitUntilCompleted];

        if (lastComputeCB.status == MTLCommandBufferStatusError) {
            std::fprintf(stderr, "FATAL: final command buffer completed with an error: %s\n",
                          lastComputeCB.error.localizedDescription.UTF8String);
            return 1;
        }

        std::fprintf(stderr, "Pipeline complete. Verifying output against CPU reference...\n");
        const float* output_ptr = static_cast<const float*>(output.contents);
        size_t total = NUM_SLICES * ELEMENTS_PER_SLICE;
        size_t mismatches = 0;
        const float kTolerance = 1e-3f;
        for (size_t k = 0; k < total; ++k) {
            float diff = std::fabs(output_ptr[k] - cpu_reference[k]);
            if (diff > kTolerance) {
                if (mismatches < 5) {
                    std::fprintf(stderr, "  MISMATCH at index %zu: got %f, expected %f\n",
                                  k, output_ptr[k], cpu_reference[k]);
                }
                ++mismatches;
            }
        }

        if (mismatches == 0) {
            std::fprintf(stderr, "PASS: all %zu output values match the CPU reference "
                                  "(tolerance %g).\n", total, kTolerance);
            return 0;
        } else {
            std::fprintf(stderr, "FAIL: %zu / %zu values mismatched.\n", mismatches, total);
            return 1;
        }
    }
}