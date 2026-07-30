add_test([=[RingBuffer.PushPopSingleThreadedFIFO]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=RingBuffer.PushPopSingleThreadedFIFO]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RingBuffer.PushPopSingleThreadedFIFO]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_ring_buffer.cpp:12]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[RingBuffer.WrapAroundReuse]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=RingBuffer.WrapAroundReuse]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RingBuffer.WrapAroundReuse]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_ring_buffer.cpp:29]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[RingBuffer.SizeApproxTracksPushesAndPops]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=RingBuffer.SizeApproxTracksPushesAndPops]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RingBuffer.SizeApproxTracksPushesAndPops]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_ring_buffer.cpp:39]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[RingBuffer.SPMCStressNoDropsNoDuplicates]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=RingBuffer.SPMCStressNoDropsNoDuplicates]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RingBuffer.SPMCStressNoDropsNoDuplicates]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_ring_buffer.cpp:54]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[RingBuffer.CapacityReportedCorrectly]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=RingBuffer.CapacityReportedCorrectly]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RingBuffer.CapacityReportedCorrectly]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_ring_buffer.cpp:106]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[PageTable.AllocatesBlocksLazilyAsSequenceGrows]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=PageTable.AllocatesBlocksLazilyAsSequenceGrows]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[PageTable.AllocatesBlocksLazilyAsSequenceGrows]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_page_table.cpp:10]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[PageTable.FreeSequenceReturnsBlocksToPool]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=PageTable.FreeSequenceReturnsBlocksToPool]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[PageTable.FreeSequenceReturnsBlocksToPool]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_page_table.cpp:30]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[PageTable.FreeUnknownSequenceIsNoOp]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=PageTable.FreeUnknownSequenceIsNoOp]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[PageTable.FreeUnknownSequenceIsNoOp]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_page_table.cpp:40]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[PageTable.ExhaustionReturnsFalseWithoutPartialAllocation]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=PageTable.ExhaustionReturnsFalseWithoutPartialAllocation]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[PageTable.ExhaustionReturnsFalseWithoutPartialAllocation]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_page_table.cpp:46]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[PageTable.NoBlockSharedBetweenTwoSequences]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=PageTable.NoBlockSharedBetweenTwoSequences]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[PageTable.NoBlockSharedBetweenTwoSequences]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_page_table.cpp:58]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[PageTable.FragmentationRatioComputation]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=PageTable.FragmentationRatioComputation]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[PageTable.FragmentationRatioComputation]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_page_table.cpp:73]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[PageTable.BlocksNeededForRounding]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=PageTable.BlocksNeededForRounding]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[PageTable.BlocksNeededForRounding]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_page_table.cpp:82]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[PageTable.ConcurrentGrowAndFreeIsRaceFree]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=PageTable.ConcurrentGrowAndFreeIsRaceFree]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[PageTable.ConcurrentGrowAndFreeIsRaceFree]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_page_table.cpp:94]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[Scheduler.SinglePromptGoesPrefillThenDecodeThenFinished]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=Scheduler.SinglePromptGoesPrefillThenDecodeThenFinished]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Scheduler.SinglePromptGoesPrefillThenDecodeThenFinished]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_scheduler.cpp:18]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[Scheduler.LargePromptSpansMultiplePrefillChunks]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=Scheduler.LargePromptSpansMultiplePrefillChunks]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Scheduler.LargePromptSpansMultiplePrefillChunks]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_scheduler.cpp:45]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[Scheduler.DecodeIsPrioritizedOverNewPrefillAdmission]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=Scheduler.DecodeIsPrioritizedOverNewPrefillAdmission]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Scheduler.DecodeIsPrioritizedOverNewPrefillAdmission]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_scheduler.cpp:68]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[Scheduler.BlockedOnPageTableCapacityDoesNotCrashOrCorrupt]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=Scheduler.BlockedOnPageTableCapacityDoesNotCrashOrCorrupt]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Scheduler.BlockedOnPageTableCapacityDoesNotCrashOrCorrupt]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_scheduler.cpp:95]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
add_test([=[Scheduler.RandomizedWorkloadNeverThrowsAndAlwaysFinishesEventually]=]  /Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan/aether_tests [==[--gtest_filter=Scheduler.RandomizedWorkloadNeverThrowsAndAlwaysFinishesEventually]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[Scheduler.RandomizedWorkloadNeverThrowsAndAlwaysFinishesEventually]=]
  PROPERTIES
    
    DEF_SOURCE_LINE [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/tests/test_scheduler.cpp:129]==]
    WORKING_DIRECTORY [==[/Users/akhilarya/UWMadisonAndF1/Projects/AetherMoE/AetherMoE/build-tsan]==]
    SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==]
    
)
set(aether_tests_TESTS [==[RingBuffer.PushPopSingleThreadedFIFO]==] [==[RingBuffer.WrapAroundReuse]==] [==[RingBuffer.SizeApproxTracksPushesAndPops]==] [==[RingBuffer.SPMCStressNoDropsNoDuplicates]==] [==[RingBuffer.CapacityReportedCorrectly]==] [==[PageTable.AllocatesBlocksLazilyAsSequenceGrows]==] [==[PageTable.FreeSequenceReturnsBlocksToPool]==] [==[PageTable.FreeUnknownSequenceIsNoOp]==] [==[PageTable.ExhaustionReturnsFalseWithoutPartialAllocation]==] [==[PageTable.NoBlockSharedBetweenTwoSequences]==] [==[PageTable.FragmentationRatioComputation]==] [==[PageTable.BlocksNeededForRounding]==] [==[PageTable.ConcurrentGrowAndFreeIsRaceFree]==] [==[Scheduler.SinglePromptGoesPrefillThenDecodeThenFinished]==] [==[Scheduler.LargePromptSpansMultiplePrefillChunks]==] [==[Scheduler.DecodeIsPrioritizedOverNewPrefillAdmission]==] [==[Scheduler.BlockedOnPageTableCapacityDoesNotCrashOrCorrupt]==] [==[Scheduler.RandomizedWorkloadNeverThrowsAndAlwaysFinishesEventually]==])
