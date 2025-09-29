#include <iostream>
#include "ad/ag_all.hpp"

using namespace ag;

int main() {
    // Build a tiny graph: y = (x @ W) + b
    Value x  = constant(Tensor::randn(2,2,123), "x");
    Value W  = param(Tensor::randn(2,2,321), "W");
    Value b  = param(Tensor::zeros(1,2), "b");

    // Separate matmul so we can checkpoint it
    Value M = matmul(x, W);   // checkpoint candidate
    Value y = M + b;          // final output

    std::cout << "=== Initial graph ===\n";
    std::cout << "y.node->inputs.size() = " << y.node->inputs.size() << "\n";
    std::cout << "M.node->inputs.size() = " << M.node->inputs.size() << "\n";

    // -----------------------
    // Case A: careful deletion only
    // -----------------------
    ag::delete_subgraph(y.node.get());

    std::cout << "\n=== After delete_subgraph(y) ===\n";
    std::cout << "y.node->inputs.size() = " << y.node->inputs.size() << " (should be 0)\n";
    std::cout << "M.node->inputs.size() = " << M.node->inputs.size() << " (may be 0 if freed)\n";

    // Rebuild graph fresh for checkpoint test
    Value x2  = constant(Tensor::randn(2,2,123), "x2");
    Value W2  = param(Tensor::randn(2,2,321), "W2");
    Value b2  = param(Tensor::zeros(1,2), "b2");
    Value M2 = matmul(x2, W2);
    Value y2 = M2 + b2;

    // -----------------------
    // Case B: register checkpoint, then delete
    // -----------------------
    ag::checkpoint_register(M2.node.get());   // keep matmul alive
    ag::delete_subgraph_preserve_checkpoints(y2.node.get());

    std::cout << "\n=== After delete_subgraph_preserve_checkpoints(y2) with M2 checkpointed ===\n";
    std::cout << "y2.node->inputs.size() = " << y2.node->inputs.size() << "\n";
    if (!y2.node->inputs.empty()) {
        std::cout << " y2.input[0] valid? " << std::boolalpha << (bool)y2.node->inputs[0] << "\n";
    }
    std::cout << "M2.node->inputs.size() = " << M2.node->inputs.size()
              << " (should still have inputs, since checkpointed)\n";

    // cleanup
    ag::checkpoint_clear_all();
    return 0;
}
