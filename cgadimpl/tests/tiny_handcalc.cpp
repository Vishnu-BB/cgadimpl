// =====================
// file: tests/tiny_handcalc.cpp
// =====================
#include "ad/graph.hpp"
#include "ad/ops.hpp"
#include "ad/autodiff.hpp"
#include <iostream>
#include <cmath>

using namespace ag;

int main() {
    std::cout << "=== Tiny hand-checkable MLP (2x2) ===\n";

    // Shapes
    const int B = 2;   // batch
    const int In = 2;  // input dim
    const int H  = 2;  // hidden dim
    const int Out= 2;  // classes

    // -------- Data (X) --------
    // X = [[ 1, -1 ],
    //      [ 0,  2 ]]
    Tensor Xt(B, In);
    Xt(0,0)= 1; Xt(0,1)=-1;
    Xt(1,0)= 0; Xt(1,1)= 2;
    Value X = constant(Xt, "X");
    std::cout << "X:\n" << Xt << "\n";

    // One-hot labels Y = [[1,0],[0,1]]
    Tensor Yt(B, Out);
    Yt(0,0)=1; Yt(0,1)=0;
    Yt(1,0)=0; Yt(1,1)=1;
    Value Y = constant(Yt, "Y");
    std::cout << "Y (one-hot):\n" << Yt << "\n";

    // -------- Layer 1 params (W1, b1) --------
    // W1 = [[ 2, 0 ],
    //       [-1, 1 ]]
    Tensor W1t(In, H);
    W1t(0,0)= 2; W1t(0,1)=0;
    W1t(1,0)=-1; W1t(1,1)=1;
    Value W1 = param(W1t, "W1");

    // b1 = [[ 1, -2 ]]
    Tensor b1t(1, H);
    b1t(0,0)= 1; b1t(0,1)=-2;
    Value b1 = param(b1t, "b1");

    std::cout << "W1:\n" << W1t << "b1:\n" << b1t << "\n";

    // -------- Forward: L1 = ReLU(X@W1 + b1) --------
    Value Z1   = matmul(X, W1);     // [B,H]
    Value Z1_b = Z1 + b1;           // broadcast b1 over rows
    Value L1   = relu(Z1_b);

    std::cout << "Z1 = X @ W1:\n"       << Z1.val();
    std::cout << "Z1_b = Z1 + b1:\n"    << Z1_b.val();
    std::cout << "L1 = ReLU(Z1_b):\n"   << L1.val() << "\n";

    // -------- Layer 2 params (W2, b2) --------
    // W2 = [[ 1, -1 ],
    //       [ 2,  0 ]]
    Tensor W2t(H, Out);
    W2t(0,0)=1; W2t(0,1)=-1;
    W2t(1,0)=2; W2t(1,1)= 0;
    Value W2 = param(W2t, "W2");

    // b2 = [[ 0, 1 ]]
    Tensor b2t(1, Out);
    b2t(0,0)=0; b2t(0,1)=1;
    Value b2 = param(b2t, "b2");

    std::cout << "W2:\n" << W2t << "b2:\n" << b2t << "\n";

    // -------- Forward: logits, softmax, loss --------
    Value logits = matmul(L1, W2) + b2;        // [B,Out]
    std::cout << "logits = L1 @ W2 + b2:\n" << logits.val();

    Value probs  = softmax_row(logits);        // [B,Out]
    std::cout << "probs = softmax_row(logits):\n" << probs.val();

    Value loss   = cross_entropy_with_logits(logits, Y); // [1,1]
    std::cout << "loss (CE with logits):\n" << loss.val() << "\n";

    // -------- Backward: just to show grads exist --------
    backward(loss);
    std::cout << "grad W1:\n" << W1.node->grad;
    std::cout << "grad b1:\n" << b1.node->grad;
    std::cout << "grad W2:\n" << W2.node->grad;
    std::cout << "grad b2:\n" << b2.node->grad << "\n";

    // -------- Hand-check notes (expected by manual calc) --------
    // Z1 = X@W1 =
    //   row0: [1*2 + (-1)*(-1), 1*0 + (-1)*1] = [3, -1]
    //   row1: [0*2 + 2*(-1),    0*0 + 2*1   ] = [-2, 2]
    // Z1_b = Z1 + b1 =
    //   row0: [3+1,  -1-2] = [ 4, -3]
    //   row1: [-2+1,  2-2] = [-1,  0]
    // L1 = ReLU(Z1_b) = [[4,0],[0,0]]
    //
    // logits = L1@W2 + b2 =
    //   row0: [4*1 + 0*2, 4*(-1)+0*0] + [0,1] = [4, -3] + [0,1] = [4, -2]
    //   row1: [0,0] + [0,1] = [0,1]
    // probs(row0) ~ softmax([4,-2]) = [exp(4),exp(-2)]/sum ≈ [0.9975, 0.0025]
    // probs(row1) ~ softmax([0,1])  = [0.2689, 0.7311]
    // Y = [[1,0],[0,1]]
    // CE per row = -log p(correct); mean over 2 rows.
    //
    // You can check that printed tensors match these numbers.

    std::cout << "=== Done: tiny hand-checkable test ===\n";
    return 0;
}
