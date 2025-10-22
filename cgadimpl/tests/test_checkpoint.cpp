// // ============================================================
// // File: test_checkpoint.cpp
// // Purpose: Verify gradient checkpointing and recomputation
// // ============================================================

// #include <iostream>
// #include <vector>
// #include "ad/ag_all.hpp"
// #include "ad/ops.hpp"
// #include "ad/kernels_api.hpp"

// using namespace ag;

// int main() {
//     std::cout << "===== Gradient Checkpointing Test =====\n";
//     ag::kernels::load_cpu_plugin("./libagkernels_cpu.so");
//     // 1. Create some simple input tensors
//     Tensor x_data = Tensor::randn(2, 2, 42);  // small deterministic input
//     Tensor W_data = Tensor::randn(2, 2, 123);
//     Tensor b_data = Tensor::randn(2, 2, 7);

//     // 2. Wrap them as Values for the computational graph
//     Value x = constant(x_data, "x");
//     Value W = param(W_data, "W");
//     Value b = param(b_data, "b");

//     // 3. Build a small network with checkpointed middle layer
//     //    y = ((x @ W) + b).relu()
//     Value y1 = matmul(x, W);
//     Value y2 = add(y1, b);

//     // Mark y2 as a checkpoint
//     y2 = inplace_checkpoint(y2);

//     // Apply activation
//     Value y3 = relu(y2);
//     Value loss = sum(y3);  // simple scalar loss

//     // 4. Backward pass
//     backward(loss);

//     // 5. Verify that checkpointed nodes recompute
//     std::cout << "\n--- Checkpoint verification ---\n";
//     auto n = y2.node;
//     if (n->is_checkpoint) {
//         std::cout << "Node " << n->debug_name << " is checkpointed ✅\n";
//     } else {
//         std::cout << "Node " << n->debug_name << " is NOT checkpointed ❌\n";
//     }

//     // 6. Inspect gradient values
//     std::cout << "\nGradients:\n";
//     std::cout << "dL/dW:\n" << W.grad() << "\n";
//     std::cout << "dL/db:\n" << b.grad() << "\n";

//     // 7. Check recomputation correctness manually
//     std::cout << "\nRecomputing checkpoint manually...\n";
//     bool recomputed = checkpoint_impl::recompute_subgraph(y2.node->shared_from_this());
//     std::cout << (recomputed ? "Recomputation success ✅\n" : "Recomputation failed ❌\n");

//     // 8. Print recomputed value
//     std::cout << "\nCheckpointed node value after recompute:\n";
//     std::cout << y2.node->value << "\n";

//     std::cout << "===== Test completed successfully =====\n";
//     return 0;
// }

// ============================================================
// File: test_auto_checkpoint.cpp
// Purpose: Verify automatic gradient checkpointing (every_n & by_depth)
// ============================================================

// #include <iostream>
// #include <vector>
// #include "ad/ag_all.hpp"
// #include "ad/checkpoint.hpp"
// #include "ad/kernels_api.hpp"
// #include <unordered_set>
// #include <deque>

// using namespace ag;

// int main() {
//     std::cout << "===== Auto Gradient Checkpointing Test =====\n";
//     // ag::kernels::load_cpu_plugin("./libagkernels_cpu.so");
//     // ------------------------------------------------------------
//     // 1. Prepare small deterministic tensors
//     Tensor x_data = Tensor::randn(2, 4, 42);
//     Tensor W1_data = Tensor::randn(4, 4, 123);
//     Tensor W2_data = Tensor::randn(4, 4, 321);
//     Tensor W3_data = Tensor::randn(4, 2, 999);
//     Tensor b1_data = Tensor::randn(1, 4, 55);
//     Tensor b2_data = Tensor::randn(1, 4, 77);
//     Tensor b3_data = Tensor::randn(1, 2, 88);

//     // ------------------------------------------------------------
//     // 2. Wrap them as Values
//     Value x = constant(x_data, "x");
//     Value W1 = param(W1_data, "W1");
//     Value W2 = param(W2_data, "W2");
//     Value W3 = param(W3_data, "W3");
//     Value b1 = param(b1_data, "b1");
//     Value b2 = param(b2_data, "b2");
//     Value b3 = param(b3_data, "b3");

//     // ------------------------------------------------------------
//     // 3. Build a deeper network
//     // y = relu((relu((x @ W1 + b1) @ W2 + b2)) @ W3 + b3)
//     Value h1 = relu(add(matmul(x, W1), b1));
//     Value h2 = relu(add(matmul(h1, W2), b2));
//     Value y = add(matmul(h2, W3), b3);
//     Value loss = sum(relu(y));  // scalar loss

//     // ------------------------------------------------------------
//     // 4. Apply automatic checkpointing
//     std::cout << "\nApplying auto checkpointing...\n";
//     auto_checkpoint_every_n(loss, 2);       // mark every 2nd node
//     auto_checkpoint_by_depth(loss, 3);      // mark nodes deeper than depth 3

//     // ------------------------------------------------------------
//     // 5. Verify which nodes got checkpointed
//     std::cout << "\n--- Auto checkpoint verification ---\n";
//     int checkpointed_count = 0;
//     std::deque<std::shared_ptr<Node>> q;
//     std::unordered_set<Node*> visited;
//     q.push_back(loss.node);

//     while (!q.empty()) {
//         auto n = q.front(); q.pop_front();
//         if (!n || visited.count(n.get())) continue;
//         visited.insert(n.get());
//         if (n->is_checkpoint) {
//             ++checkpointed_count;
//             std::cout << "Checkpointed node: " << n->debug_name << " ✅\n";
//         }
//         for (auto &p : n->inputs)
//             if (p) q.push_back(p);
//     }

//     if (checkpointed_count == 0)
//         std::cout << "❌ No nodes were marked as checkpointed.\n";
//     else
//         std::cout << "✅ Total checkpointed nodes: " << checkpointed_count << "\n";

//     // ------------------------------------------------------------
//     // 6. Backward pass (triggers recomputation of checkpointed nodes)
//     backward(loss);

//     // ------------------------------------------------------------
//     // 7. Inspect gradients for parameters
//     std::cout << "\nGradients:\n";
//     std::cout << "dL/dW1:\n" << W1.grad() << "\n";
//     std::cout << "dL/dW2:\n" << W2.grad() << "\n";
//     std::cout << "dL/dW3:\n" << W3.grad() << "\n";
//     std::cout << "dL/db3:\n" << b3.grad() << "\n";

//     // ------------------------------------------------------------
//     // 8. Manual recomputation test on one of the checkpointed nodes
//     std::cout << "\nManual recompute verification:\n";
//     for (auto &n : visited) {
//     if (n->is_checkpoint && !n->inputs.empty()) {
//         bool ok = checkpoint_impl::recompute_subgraph(n->shared_from_this());
//         std::cout << "Recomputed node (" << n->debug_name << "): "
//                   << (ok ? "✅" : "❌") << "\n";
//             break;
//         }
//     }


//     std::cout << "\n===== Auto Checkpoint Test Completed =====\n";
//     return 0;
// }

// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <iomanip>
// #include <filesystem>
// #include "ad/ag_all.hpp"
// #include "ad/ops.hpp"
// #include "ad/kernels_api.hpp"
// #include "ad/checkpoint.hpp"
// #include "ad/inplace.hpp"
// #include "tensor.hpp"

// using namespace ag;

// static void print_tensor(const std::string& name, const Tensor& t, int max = 4) {
//     std::cout << name << " [" << t.rows() << "x" << t.cols() << "]: ";
//     const float* ptr = t.data();         // ✅ use public accessor
//     int n = t.size();
//     for (int i = 0; i < std::min(n, max); ++i)
//         std::cout << std::fixed << std::setprecision(4) << ptr[i] << " ";
//     if (n > max) std::cout << "...";
//     std::cout << "\n";
// }


// int main() {
//     std::cout << "===== Complex Gradient Checkpointing Test =====\n";

//     // ------------------------------------------------------------------------
//     // 1. Load optimized CPU kernels plugin (forward + backward)
//     // ------------------------------------------------------------------------
//     ag::kernels::load_cpu_plugin("./libagkernels_cpu.so");
//     // std::cout << "✅ Loaded optimized CPU kernels plugin.\n";

//     // ------------------------------------------------------------------------
//     // 2. Construct deterministic data
//     // ------------------------------------------------------------------------
//     const int B = 8, In = 16, H1 = 64, H2 = 64, Out = 8;
//     Tensor x_t  = Tensor::randn(B, In, 42);
//     Tensor W1_t = Tensor::randn(In, H1, 123);
//     Tensor b1_t = Tensor::zeros(1, H1);
//     Tensor W2_t = Tensor::randn(H1, H2, 321);
//     Tensor b2_t = Tensor::zeros(1, H2);
//     Tensor W3_t = Tensor::randn(H2, Out, 999);
//     Tensor b3_t = Tensor::zeros(1, Out);

//     // Wrap as Values
//     Value X  = constant(x_t, "X");
//     Value W1 = param(W1_t, "W1"), b1 = param(b1_t, "b1");
//     Value W2 = param(W2_t, "W2"), b2 = param(b2_t, "b2");
//     Value W3 = param(W3_t, "W3"), b3 = param(b3_t, "b3");

//     // ------------------------------------------------------------------------
//     // 3. Forward (baseline, no checkpoint)
//     // ------------------------------------------------------------------------
//     auto forward_mlp = [&](bool checkpoint) -> Value {
//         std::cout << "Running matmul(X, W1)" << std::endl;
//         Value h1 = add(matmul(X, W1), b1);
//         if (checkpoint) h1 = inplace_checkpoint(h1);
//         h1 = relu(h1);
//         std::cout << "Running add(y1, b1)" << std::endl;
//         Value h2 = add(matmul(h1, W2), b2);
//         if (checkpoint) h2 = inplace_checkpoint(h2);
//         h2 = relu(h2);

//         Value out = add(matmul(h2, W3), b3);
//         if (checkpoint) out = inplace_checkpoint(out);
//         return relu(out);
//     };

//     std::cout << "\n--- Baseline (no checkpoint) ---\n";
//     Value y_normal = forward_mlp(false);
//     Value loss_normal = sum(y_normal);

//     auto t0 = std::chrono::high_resolution_clock::now();
//     zero_grad(loss_normal);
//     backward(loss_normal);
//     auto t1 = std::chrono::high_resolution_clock::now();
//     double base_time = std::chrono::duration<double, std::milli>(t1 - t0).count();

//     print_tensor("Loss (no checkpoint)", loss_normal.val());

//     // Save grads for later comparison
//     Tensor gradW1_base = W1.grad(), gradW2_base = W2.grad(), gradW3_base = W3.grad();

//     // ------------------------------------------------------------------------
//     // 4. Forward (with checkpoint)
//     // ------------------------------------------------------------------------
//     std::cout << "\n--- With inplace checkpoint ---\n";
//     Value y_chk = forward_mlp(true);
//     Value loss_chk = sum(y_chk);

//     t0 = std::chrono::high_resolution_clock::now();
//     zero_grad(loss_chk);
//     backward(loss_chk);
//     t1 = std::chrono::high_resolution_clock::now();
//     double chk_time = std::chrono::duration<double, std::milli>(t1 - t0).count();

//     print_tensor("Loss (with checkpoint)", loss_chk.val());

//     // ------------------------------------------------------------------------
//     // 5. Compare gradients
//     // ------------------------------------------------------------------------
//     auto diff = [](const Tensor& A, const Tensor& B) {
//         const float* a = A.data();
//         const float* b = B.data();
//         int n = std::min(A.size(), B.size());
//         double err = 0.0;
//         for (int i = 0; i < n; ++i)
//             err += std::abs(a[i] - b[i]);
//         return err / n;
//     };


//     std::cout << "\n--- Gradient Comparison ---\n";
//     std::cout << "avg|dW1_base - dW1_chk| = " << diff(gradW1_base, W1.grad()) << "\n";
//     std::cout << "avg|dW2_base - dW2_chk| = " << diff(gradW2_base, W2.grad()) << "\n";
//     std::cout << "avg|dW3_base - dW3_chk| = " << diff(gradW3_base, W3.grad()) << "\n";

//     // ------------------------------------------------------------------------
//     // 6. Timing summary
//     // ------------------------------------------------------------------------
//     std::cout << "\n--- Timing ---\n";
//     std::cout << "No checkpoint : " << base_time << " ms\n";
//     std::cout << "With checkpoint : " << chk_time << " ms\n";
//     std::cout << "Speed ratio (chk/base) = " << (chk_time / base_time) << "\n";

//     // ------------------------------------------------------------------------
//     // 7. Verify recomputation
//     // ------------------------------------------------------------------------
//     std::cout << "\n--- Manual recomputation test ---\n";
//     auto n2 = y_chk.node->inputs.front(); // last checkpointed node
//     bool recomputed = checkpoint_impl::recompute_subgraph(n2);
//     std::cout << (recomputed ? "Recomputation success ✅" : "Recomputation failed ❌") << "\n";
//     print_tensor("Recomputed value sample", n2->value);

//     // ------------------------------------------------------------------------
//     // 8. Sanity: show grads
//     // ------------------------------------------------------------------------
//     print_tensor("W1.grad (chk)", W1.grad());
//     print_tensor("W2.grad (chk)", W2.grad());
//     print_tensor("W3.grad (chk)", W3.grad());

//     std::cout << "\n===== Complex Checkpointing Test Finished =====\n";
//     return 0;
// }


#include <iostream>
#include <vector>
#include <deque>
#include <unordered_set>
#include <chrono>
#include <thread>
#include <iomanip>
#include <memory>
#include <algorithm>
#include "ad/ag_all.hpp"
#include "ad/ops.hpp"
#include "ad/checkpoint.hpp"
#include "ad/inplace.hpp"

using namespace ag;
using namespace checkpoint_impl;
// -----------------------------------------------------------
// Memory estimation helper
// -----------------------------------------------------------
struct GraphMem {
    size_t value_bytes = 0;
    size_t grad_bytes  = 0;
};
GraphMem estimate_graph_memory(const Value &root) {
    GraphMem s{};
    std::unordered_set<Node*> seen;
    std::deque<std::shared_ptr<Node>> q;
    if (!root.node) return s;
    q.push_back(root.node);
    while (!q.empty()) {
        auto n = q.front(); q.pop_front();
        if (!n || seen.count(n.get())) continue;
        seen.insert(n.get());
        s.value_bytes += n->value.numel() * sizeof(float);
        s.grad_bytes  += n->grad.numel()  * sizeof(float);
        for (auto &p : n->inputs)
            if (p) q.push_back(p);
    }
    return s;
}
double toMB(size_t b) { return double(b)/(1024.0*1024.0); }
// -----------------------------------------------------------
// Build and run a deep MLP once
// -----------------------------------------------------------
struct RunResult {
    double forward_mem_before, forward_mem_after, backward_time_ms;
};
RunResult run_network(int batch, int input_dim, int hidden, int depth,
                      int checkpoint_interval, bool use_ckpt)
{
    // 1. Build data + params
    Tensor x_data = Tensor::randn(batch, input_dim, 123);
    Value x = constant(x_data, "x");

    std::vector<Value> W, b;
    for (int l=0;l<depth;++l){
        Tensor Wt = Tensor::randn((l==0)?input_dim:hidden, hidden, 100+l);
        Tensor bt = Tensor::randn(1, hidden, 200+l);
        W.push_back(param(Wt, ("W"+std::to_string(l)).c_str()));
        b.push_back(param(bt, ("b"+std::to_string(l)).c_str()));
    }
    Tensor Wout_t = Tensor::randn(hidden, input_dim, 300);
    Tensor bout_t = Tensor::randn(1, input_dim, 400);
    Value Wout = param(Wout_t, "Wout");
    Value bout = param(bout_t, "bout");

    // 2. Build forward graph
    Value cur = x;
    for (int l=0;l<depth;++l){
        cur = matmul(cur, W[l]);
        cur = add(cur, b[l]);
        if (use_ckpt && ((l+1)%checkpoint_interval)==1)
            cur = checkpoint(cur, CheckpointOptions());
        cur = relu(cur);
    }
    cur = matmul(cur, Wout);
    cur = add(cur, bout);
    Value loss = sum(cur); // scalar
    // std::cout<<loss.shape().first<<"  Lets enjoy now, time is precious   "<<loss.shape().second;

    // 3. Memory before eviction (peak forward)
    GraphMem m_before = estimate_graph_memory(loss);

    // 4. Evict activations for checkpoint version
    GraphMem m_after;
    if (use_ckpt) {
        evict_non_checkpoint_values(loss);
        m_after = estimate_graph_memory(loss);
    } else {
        m_after = m_before;
    }

    // 5. Backward timing (triggers recompute)
    auto t0 = std::chrono::high_resolution_clock::now();
    backward(loss);
    auto t1 = std::chrono::high_resolution_clock::now();
    double t_ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    return { toMB(m_before.value_bytes), toMB(m_after.value_bytes), t_ms };
}

// -----------------------------------------------------------
// Benchmark runner
// -----------------------------------------------------------
int main(){
    std::cout << "===== Gradient Checkpointing: Memory vs Time =====\n";
    // Large enough network to show a clear difference
    int batch = 8;
    int input_dim = 16;
    int hidden = 256;
    int depth  = 64;
    int ckpt_interval = 3;

    std::cout << "Config: batch="<<batch<<" input_dim="<<input_dim
              <<" hidden="<<hidden<<" depth="<<depth
              <<" checkpoint_interval="<<ckpt_interval<<"\n\n";

    std::cout << "[Run 1] Without checkpointing...\n";
    auto no = run_network(batch,input_dim,hidden,depth,ckpt_interval,false);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cout << "\n[Run 2] With checkpointing...\n";
    auto ck = run_network(batch,input_dim,hidden,depth,ckpt_interval,true);

    // -------------------------------------------------------
    // Report
    // -------------------------------------------------------
    std::cout<<std::fixed<<std::setprecision(3);
    std::cout<<"\n--- RESULTS ---\n";
    std::cout<<"Without checkpointing:\n";
    std::cout<<"  Forward activations: "<<no.forward_mem_before<<" MB\n";
    std::cout<<"  Backward time:       "<<no.backward_time_ms<<" ms\n\n";
    std::cout<<"With checkpointing:\n";
    std::cout<<"  Forward activations (before eviction): "<<ck.forward_mem_before<<" MB\n";
    std::cout<<"  After eviction:                       "<<ck.forward_mem_after<<" MB\n";
    std::cout<<"  Backward time (with recompute):       "<<ck.backward_time_ms<<" ms\n\n";

    double mem_saved = 100.0*(1.0-ck.forward_mem_after/no.forward_mem_before);
    double time_over = 100.0*(ck.backward_time_ms/no.backward_time_ms - 1.0);
    std::cout<<"Memory saved: "<<mem_saved<<" %\n";
    std::cout<<"Recompute overhead: "<<time_over<<" %\n";

    std::cout<<"\nInterpretation:\n";
    std::cout<<"  • Expect memory_after < memory_before for checkpointed run.\n";
    std::cout<<"  • Expect backward_time_with_ckpt > without_ckpt.\n";
    std::cout<<"  • More depth or smaller checkpoint_interval increases both effects.\n";
    std::cout<<"============================================\n";
    return 0;
}
