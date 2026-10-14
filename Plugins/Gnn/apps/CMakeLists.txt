# Standalone throughput test of the GNN track finding on traccc event data
add_executable(ActsGnnTracccThroughput TracccGnnThroughput.cpp)
target_link_libraries(
    ActsGnnTracccThroughput
    PRIVATE
        ActsPluginGnn
        traccc::core
        traccc::cuda
        traccc::io
        traccc::device_common
        vecmem::core
        vecmem::cuda
        CUDA::cudart
        Threads::Threads
)
set_target_properties(
    ActsGnnTracccThroughput
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
install(TARGETS ActsGnnTracccThroughput RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
