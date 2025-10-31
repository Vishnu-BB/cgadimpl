

// tests/test_checkpoint_memory.cpp
// Measures memory saving from your checkpoint implementation.
//
// Build like your other tests and run: ./test_checkpoint_memory
// Requires: ad/ag_all.hpp, ad/checkpoint.hpp, ad/inplace.hpp, ops etc.

#include <iostream>
#include <vector>
#include <deque>
#include <unordered_set>
#include <iomanip>
#include <chrono>
#include "ad/ag_all.hpp"
#include "ad/ops.hpp"
#include "ad/checkpoint.hpp"
#include "ad/inplace.hpp"

using namespace ag;
using namespace checkpoint_impl;

// -----------------------------------------------
// Small feedforward graph (MLP-like)
// -----------------------------------------------
Value build_tiny_network(int batch = 4, int in_dim = 8, int hidden = 16, int depth = 4) {
    Tensor x_data = Tensor::randn(batch, in_dim, 123);
    Value x = constant(x_data, "x");

    Value cur = x;
    for (int i = 0; i < depth; ++i) {
        Tensor Wt = Tensor::randn((i == 0 ? in_dim : hidden), hidden, 100 + i);
        Tensor bt = Tensor::randn(1, hidden, 200 + i);
        Value W = param(Wt, ("W" + std::to_string(i)).c_str());
        Value b = param(bt, ("b" + std::to_string(i)).c_str());
        cur = add(matmul(cur, W), b);
        cur = relu(cur);

        // Mark every 2nd layer as a checkpoint
        if (i % 2 == 0) {
            mark_node_checkpoint(cur.node, CheckpointOptions());
        }
    }

    Tensor Wout_t = Tensor::randn(hidden, 1, 300);
    Tensor bout_t = Tensor::randn(1, 1, 400);
    Value Wout = param(Wout_t, "Wout");
    Value bout = param(bout_t, "bout");

    Value out = add(matmul(cur, Wout), bout);
    Value loss = sum(out);
    return loss;
}

// -----------------------------------------------
// Memory estimation utility
// -----------------------------------------------
size_t estimate_bytes(const Value &root) {
    std::unordered_set<Node*> seen;
    std::deque<std::shared_ptr<Node>> q;
    if (!root.node) return 0;
    q.push_back(root.node);
    size_t bytes = 0;
    while (!q.empty()) {
        auto n = q.front(); q.pop_front();
        if (!n || seen.count(n.get())) continue;
        seen.insert(n.get());
        bytes += n->value.numel() * sizeof(float);
        for (auto &p : n->inputs) if (p) q.push_back(p);
    }
    return bytes;
}

// -----------------------------------------------
// Pretty print activation stats
// -----------------------------------------------
void print_activation_stats(const std::string& tag, const Value &root) {
    size_t bytes = estimate_bytes(root);
    std::cout << tag << ": activations = "
              << std::fixed << std::setprecision(3)
              << (double)bytes / (1024.0 * 1024.0) << " MB\n";
}

// -----------------------------------------------
// Test: end-to-end checkpoint memory + recompute
// -----------------------------------------------
int main() {
    std::cout << "===== Checkpointing Memory & Recomputation Test =====\n";

    // Build a graph
    Value loss = build_tiny_network(8, 64, 128, 8);

    std::cout << "\n[1] Computing forward values...\n";
    compute_forward_values(loss);
    print_activation_stats("Before snapshot", loss);

    std::cout << "[2] Capturing checkpoint snapshots...\n";
    capture_checkpoint_snapshots(loss);

    std::cout << "[3] Evicting non-checkpoint activations...\n";
    evict_non_checkpoint_values(loss);
    print_activation_stats("After eviction", loss);

    // Measure backward time
    std::cout << "\n[4] Running backward pass (with recompute)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    backward(loss);
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[DONE] Backward completed successfully.\n";
    std::cout << "Backward time = " << t_ms << " ms\n";

    // Sanity: check memory difference
    size_t before = estimate_bytes(loss);
    std::cout << "\nFinal activations (after backward): "
              << std::fixed << std::setprecision(3)
              << (double)before / (1024.0 * 1024.0) << " MB\n";

    std::cout << "===== Test complete =====\n";
    return 0;
}
