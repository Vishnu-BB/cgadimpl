#include <iostream>
#include "ad/ag_all.hpp"
#include "ad/tracer.hpp"

using namespace ag;

// Helper: convert Op enum to readable string (adjust names as in your Op enum)
static const char* op_to_str(Op op) {
    switch (op) {
        case Op::Leaf: return "Leaf";
        case Op::MatMul: return "Matmul";
        case Op::Add: return "Add";
        case Op::Mul: return "Mul";
        case Op::GELU: return "Gelu";
        case Op::MSELoss: return "MseLoss";
        case Op::Relu: return "Relu";
        // case Op::Softmax: return "Softmax";
        case Op::CeWithLogits: return "CrossEntropy";
        default: return "Unknown";
    }
}

int main() {
    std::cout << "=== Graph Capture / Tracer Test ===\n";

    auto tracer = trace::make_tracer();

    {
        trace::CaptureGuard guard(tracer);

        const int B = 256;
        const int In = 512;
        const int H = 1024;
        const int Out = 256;

        // --- Inputs ---
        Tensor Xt = Tensor::randn(B, In, 123);
        Value X = constant(Xt, "X");

        auto W1 = param(Tensor::randn(In, H, 1001), "W1");
        auto b1 = param(Tensor::zeros(1, H), "b1");

        auto W2 = param(Tensor::randn(H, Out, 1002), "W2");
        auto b2 = param(Tensor::zeros(1, Out), "b2");

        // --- Forward ---
        Value L1 = gelu(matmul(X, W1) + b1);
        Value logits = matmul(L1, W2) + b2;

        Tensor Yt = Tensor::randn(B, Out, 2001);
        Value Y = constant(Yt, "Y");

        Value loss = mse_loss(logits, Y);

        // mark loss node as output
        tracer->mark_output(loss.node);
    }

    auto topo = tracer->topo_sort();
    std::cout << "Captured " << topo.size() << " nodes:\n";

    for (size_t i = 0; i < topo.size(); ++i) {
        Node* n = topo[i].get();
        std::cout << "[" << i << "] op=" << op_to_str(n->op);
    if (n->debug_name && n->debug_name[0] != '\0')
        std::cout << " (" << n->debug_name << ")";
        std::cout << " inputs=" << n->inputs.size()
                  << " requires_grad=" << n->requires_grad << "\n";
    }

    auto outs = tracer->outputs();
    std::cout << "\nDetected outputs (" << outs.size() << "):\n";
    for (auto& o : outs) {
        std::cout << " - " << op_to_str(o->op) << "\n";
    }

    if (topo.empty()) {
        std::cerr << "❌ No nodes captured — check debug hook.\n";
        return 1;
    }

    std::cout << "✅ Tracer test passed — graph captured successfully.\n";
    return 0;
}
