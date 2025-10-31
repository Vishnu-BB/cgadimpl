// #include <iostream>
// #include <iomanip>
// #include "ad/ag_all.hpp"
// #include "optim.hpp"

// using namespace ag;

// // Utility: pretty print tensors
// void print_tensor(const std::string& name, const Tensor& t) {
//     std::cout << name << " = " << t << std::endl;
// }

// int main() {
//     std::cout << "===== Optimizer (SGD) Test =====" << std::endl;

//     // Input and target setup
//     Tensor X_data = Tensor::randn(2, 3);
//     Tensor Y_data = Tensor::randn(2, 2);

//     auto X = make_tensor(X_data, "X", false);        // input (no grad)
//     auto W = param(Tensor::randn(3, 2), "W");        // weights
//     auto bias = param(Tensor::zeros(1, 2), "bias");  // bias
//     auto target = constant(Y_data, "Y");

//     float lr = 0.1;

//     for (int epoch = 0; epoch < 5; ++epoch) {
//         zero_grad(W);
//         zero_grad(bias);

//         // Forward: simple linear + mean squared error
//         auto pred = matmul(X, W) + bias;
//         auto loss = mse_loss(pred, target);

//         // Print before backward
//         std::cout << "\nEpoch " << epoch << " =========================" << std::endl;
//         print_tensor("Loss (before)", loss.val());
//         print_tensor("W (before)", W.val());
//         print_tensor("bias (before)", bias.val());

//         // Backward
//         backward(loss);

//         // Check gradients
//         print_tensor("dL/dW", W.grad());
//         print_tensor("dL/dbias", bias.grad());

//         // Update using SGD
//         SGD(loss, nullptr, lr);

//         // After update
//         print_tensor("W (after)", W.val());
//         print_tensor("bias (after)", bias.val());
//         print_tensor("Loss (after)", loss.val());
//     }

//     std::cout << "\n✅ SGD optimizer test completed.\n";
//     return 0;
// }

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include "ad/ag_all.hpp"
#include "ad/checkpoint.hpp"
#include "ad/careful_deletion.hpp"
#include "optim.hpp"

using namespace ag;
using namespace ag::checkpoint_impl;
using namespace ag::memory;

static size_t total_value_bytes(const Value& root) {
    auto nodes = topo_from(root.node.get());
    size_t bytes = 0;
    for (auto* n : nodes)
        bytes += n->value.numel() * sizeof(float);
    return bytes;
}

static double mb(size_t bytes) { return bytes / (1024.0 * 1024.0); }

int main() {
    std::cout << "===== Optimizer (SGD) + Checkpointing Memory–Time Test =====\n";

    // -----------------------------------------------------------------
    // Model setup: simple Linear(3→2) + MSE
    // -----------------------------------------------------------------
    Tensor X_data = Tensor::randn(512, 3);     // batch 512 → larger activations
    Tensor Y_data = Tensor::randn(512, 2);

    auto X = make_tensor(X_data, "X", false);
    auto W = param(Tensor::randn(3, 2), "W");
    auto bias = param(Tensor::zeros(1, 2), "bias");
    auto target = constant(Y_data, "Y");

    float lr = 0.1f;

    // Run multiple configurations: no checkpoint vs checkpoint every N
    std::vector<int> checkpoint_stride = {0, 2}; // 0 = no checkpoint, 2 = every 2nd node
    for (int stride : checkpoint_stride) {
        std::cout << "\n=== Run with checkpoint_every_n = " << stride << " ===\n";

        // --- Forward ---
        auto pred = matmul(X, W) + bias;
        auto loss = mse_loss(pred, target);

        // Measure baseline forward activation memory
        size_t baseline_bytes = total_value_bytes(loss);
        std::cout << "[Baseline forward memory] " << std::fixed << std::setprecision(4)
                  << mb(baseline_bytes) << " MB\n";

        // --- Checkpoint marking ---
        if (stride > 0)
            auto_checkpoint_every_n(loss, stride);

        // --- Backward + recomputation timing ---
        auto start = std::chrono::high_resolution_clock::now();
        backward(loss); // your existing backward handles recompute automatically
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // --- Simulate freeing non-checkpoint activations ---
        auto nodes = topo_from(loss.node.get());
        size_t before_free = total_value_bytes(loss);
        int freed = 0;
        for (auto* n : nodes) {
            if (!n) continue;
            if (n->op == Op::Leaf) continue;
            if (n->is_checkpoint) continue;
            n->value = Tensor(); // free activation
            ++freed;
        }
        size_t after_free = total_value_bytes(loss);

        // --- Report ---
        std::cout << "Nodes freed (non-checkpoint): " << freed << "\n";
        std::cout << "[After freeing non-checkpoint activations] "
                  << mb(after_free) << " MB  (saved "
                  << (100.0 * (before_free - after_free) / before_free)
                  << "% of activations)\n";
        std::cout << "[Backward+recompute time] " << time_ms << " ms\n";

        // --- Optional: safe deletion cleanup ---
        for (auto* n : nodes) n->requires_grad = false;
        sweep_safe_nodes(loss, DeletePolicy::AlwaysSafe);
        size_t after_sweep = total_value_bytes(loss);
        std::cout << "[After careful_deletion] " << mb(after_sweep) << " MB\n";

        // --- Print summary line ---
        std::cout << std::setprecision(4);
        std::cout << "Summary:  stride=" << stride
                  << " | baseline=" << mb(baseline_bytes)
                  << " MB | after_free=" << mb(after_free)
                  << " MB | after_sweep=" << mb(after_sweep)
                  << " MB | time=" << time_ms << " ms\n";
    }

    std::cout << "\n✅ Checkpointing memory–time test completed.\n";
    return 0;
}
