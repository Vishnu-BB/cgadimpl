#include "ad/graph.hpp"  // already there
#include "ad/ag_all.hpp" // already there
#include <iostream>
#include <random>
using namespace ag;

int main() {
    // ... build model: W1,b1,...; build loss: Value loss = ...
    const int B = 8;     // batch size
    const int In = 16;   // input dim
    const int H1 = 64;
    const int H2 = 64;
    const int H3 = 32;
    const int H4 = 32;
    const int Out = 10;  // number of classes

    // ---------- Data ----------
    Tensor Xt = Tensor::randn(B, In, /*seed=*/123);
    Value  X  = constant(Xt, "X");  // inputs are constants (no grads)

    // One-hot labels Y[B,Out]
    Tensor Yt(B, Out);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> pick(0, Out - 1);
    for (int i = 0; i < B; ++i) {
        int k = pick(gen);
        for (int j = 0; j < Out; ++j) Yt(i, j) = (j == k) ? 1.f : 0.f;
    }
    Value Y = constant(Yt, "Y");

    // ---------- Parameters ----------
    // W[k][in,out], b[k][1,out]
    auto W1 = param(Tensor::randn(In,  H1, 1001), "W1");
    auto b1 = param(Tensor::zeros(1,   H1),       "b1");
    // print_value("X", X);
    // print_value("W1", W1);
    // print_value("b1", b1);

    auto W2 = param(Tensor::randn(H1,  H2, 1002), "W2");
    auto b2 = param(Tensor::zeros(1,   H2),       "b2");

    auto W3 = param(Tensor::randn(H2,  H3, 1003), "W3");
    auto b3 = param(Tensor::zeros(1,   H3),       "b3");

    auto W4 = param(Tensor::randn(H3,  H4, 1004), "W4");
    auto b4 = param(Tensor::zeros(1,   H4),       "b4");

    auto W5 = param(Tensor::randn(H4,  Out, 1005), "W5");
    auto b5 = param(Tensor::zeros(1,   Out),       "b5");

    // ---------- Forward: 4 hidden layers + logits ----------
    // L1 = GELU(X @ W1 + b1)
    Value L1 = gelu( matmul(X,  W1) + b1 );          // [B,H1]
    // print_value("L1", L1);

    // L2 = SiLU(L1 @ W2 + b2)
    Value L2 = silu( matmul(L1, W2) + b2 );          // [B,H2]
    // print_value("L2", L2);

    // L3 = LeakyReLU(L2 @ W3 + b3, alpha=0.1)
    Value L3_pre = matmul(L2, W3) + b3;              // [B,H3]
    Value L3 = leaky_relu(L3_pre, 0.1f);             // [B,H3]
    // print_value("L3", L3); 

    // L4 = Softplus(L3 @ W4 + b4)
    Value L4 = softplus( matmul(L3, W4) + b4 );      // [B,H4]
    // print_value("L4", L4);

    // Logits = L4 @ W5 + b5
    Value logits = matmul(L4, W5) + b5;              // [B,Out]
    // print_value("logits", logits);

    // Loss: stable cross-entropy with logits (one-hot)
    Value loss = cross_entropy_with_logits(logits, Y); // scalar [1,1]
    // print_value("Y (one-hot)", Y);
    // print_value("loss", loss);
    backward(loss); // eager backward to populate grads for sanity check
    // ---------- Backprop ----------

    // Tell the compiler which leaves will be passed in at runtime:
    std::vector<Value> inputs  = { X, Y };                 // data batch leaves
    std::vector<Value> params  = { W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 };  // learnable leaves

    auto comp = ag::jit::compile(loss, inputs, params);

    // Prepare pointers for run()
    std::vector<Tensor*> in_ptrs  = { &X.node->value, &Y.node->value };
    std::vector<Tensor*> par_ptrs = { &W1.node->value, &b1.node->value, &W2.node->value, &b2.node->value, &W3.node->value, &b3.node->value, &W4.node->value, &b4.node->value, &W5.node->value, &b5.node->value };

    Tensor out;                         // receives forward output (e.g., [1,1] loss)
    bool ok = comp.run(in_ptrs, par_ptrs, out);
    if (!ok) { std::cerr << "Shape guard failed; fall back to eager.\n"; return 0; }

    std::cout << "Compiled forward output:\n" << out << "\n";
//     // Sanity prints
std::cerr << "inputs: compiled=" << inputs.size() << " run=" << in_ptrs.size() << "\n";
std::cerr << "params: compiled=" << params.size() << " run=" << par_ptrs.size() << "\n";
std::cerr << "loss: eager=" << loss.val().sum_scalar() << " compiled=" << out.sum_scalar() << "\n";

    return 0;



}
