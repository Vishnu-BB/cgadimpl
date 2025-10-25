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
// #include <iostream>
// #include <vector>
// #include <deque>
// #include <unordered_set>
// #include <chrono>
// #include <thread>
// #include <iomanip>
// #include <memory>
// #include <algorithm>
// #include "ad/ag_all.hpp"
// #include "ad/ops.hpp"
// #include "ad/checkpoint.hpp"
// #include "ad/inplace.hpp"

// using namespace ag;
// using namespace checkpoint_impl;

// // ------------------------------------------------------------
// // Memory estimation helper
// // ------------------------------------------------------------
// struct GraphMem {
//     size_t value_bytes = 0;
//     size_t grad_bytes  = 0;
// };
// GraphMem estimate_graph_memory(const Value &root) {
//     GraphMem s{};
//     std::unordered_set<Node*> seen;
//     std::deque<std::shared_ptr<Node>> q;
//     if (!root.node) return s;
//     q.push_back(root.node);
//     while (!q.empty()) {
//         auto n = q.front(); q.pop_front();
//         if (!n || seen.count(n.get())) continue;
//         seen.insert(n.get());
//         s.value_bytes += n->value.numel() * sizeof(float);
//         s.grad_bytes  += n->grad.numel()  * sizeof(float);
//         for (auto &p : n->inputs)
//             if (p) q.push_back(p);
//     }
//     return s;
// }
// inline double toMB(size_t b){ return double(b)/(1024.0*1024.0); }

// // ------------------------------------------------------------
// // Transformer block builder (single layer)
// // ------------------------------------------------------------
// Value transformer_block(Value x,
//                         int embed_dim,
//                         int num_heads,
//                         std::vector<Value>& params,
//                         bool use_ckpt)
// {
//     // ---- Multi-head self-attention ----
//     int head_dim = embed_dim / num_heads;

//     Value Wq = params.at(0);
//     Value Wk = params.at(1);
//     Value Wv = params.at(2);
//     Value Wo = params.at(3);

//     Value Q = matmul(x, Wq);
//     Value K = matmul(x, Wk);
//     Value V = matmul(x, Wv);

//     Value attn_out = attention(Q, K, V, Wo);
//     if (use_ckpt) attn_out = checkpoint(attn_out, CheckpointOptions());

//     // ---- Add & Norm ----
//     Value add1 = add(x, attn_out);
//     Value norm1 = layernorm(add1);

//     if (use_ckpt) norm1 = checkpoint(norm1, CheckpointOptions());

//     // ---- Feed-forward ----
//     Value W1 = params.at(4);
//     Value b1 = params.at(5);
//     Value W2 = params.at(6);
//     Value b2 = params.at(7);

//     Value ff = relu(add(matmul(norm1, W1), b1));
//     if (use_ckpt) ff = checkpoint(ff, CheckpointOptions());

//     Value ff_out = add(matmul(ff, W2), b2);

//     // ---- Add & Norm ----
//     Value add2 = add(norm1, ff_out);
//     Value norm2 = layernorm(add2);

//     return norm2;
// }

// // ------------------------------------------------------------
// // Build full N-layer transformer stack
// // ------------------------------------------------------------
// struct RunResult { double forward_mem_before, forward_mem_after, backward_time_ms; };

// RunResult run_transformer(int batch, int seq_len, int embed_dim, int num_heads,
//                           int num_layers, int checkpoint_interval, bool use_ckpt)
// {
//     // Input tokens
//     Tensor x_data = Tensor::randn(batch*seq_len, embed_dim, 123);
//     Value x = constant(x_data, "x");

//     // Allocate parameters per layer
//     std::vector<std::vector<Value>> layer_params;
//     layer_params.reserve(num_layers);
//     for (int l = 0; l < num_layers; ++l) {
//         std::vector<Value> p;
//         int head_dim = embed_dim / num_heads;

//         // attention weights
//         p.push_back(param(Tensor::randn(embed_dim, embed_dim, 100+l), ("Wq"+std::to_string(l)).c_str()));
//         p.push_back(param(Tensor::randn(embed_dim, embed_dim, 200+l), ("Wk"+std::to_string(l)).c_str()));
//         p.push_back(param(Tensor::randn(embed_dim, embed_dim, 300+l), ("Wv"+std::to_string(l)).c_str()));
//         p.push_back(param(Tensor::randn(embed_dim, embed_dim, 400+l), ("Wo"+std::to_string(l)).c_str()));

//         // feed-forward
//         p.push_back(param(Tensor::randn(embed_dim, 4*embed_dim, 500+l), ("W1"+std::to_string(l)).c_str()));
//         p.push_back(param(Tensor::randn(1, 4*embed_dim, 600+l), ("b1"+std::to_string(l)).c_str()));
//         p.push_back(param(Tensor::randn(4*embed_dim, embed_dim, 700+l), ("W2"+std::to_string(l)).c_str()));
//         p.push_back(param(Tensor::randn(1, embed_dim, 800+l), ("b2"+std::to_string(l)).c_str()));

//         layer_params.push_back(p);
//     }

//     // Forward through transformer layers
//     Value cur = x;
//     for (int l = 0; l < num_layers; ++l) {
//         cur = transformer_block(cur, embed_dim, num_heads, layer_params[l],
//                                 use_ckpt && ((l+1) % checkpoint_interval == 0));
//     }

//     // Final loss
//     Value loss = meanall(cur);

//     // Measure memory before eviction
//     GraphMem m_before = estimate_graph_memory(loss);

//     // Evict non-checkpoints if using checkpointing
//     GraphMem m_after;
//     if (use_ckpt) {
//         evict_non_checkpoint_values(loss);
//         m_after = estimate_graph_memory(loss);
//     } else {
//         m_after = m_before;
//     }

//     // Backward
//     auto t0 = std::chrono::high_resolution_clock::now();
//     backward(loss);
//     auto t1 = std::chrono::high_resolution_clock::now();
//     double t_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

//     return { toMB(m_before.value_bytes), toMB(m_after.value_bytes), t_ms };
// }

// // ------------------------------------------------------------
// // Benchmark runner
// // ------------------------------------------------------------
// int main() {
//     std::cout << "===== Transformer Checkpointing Benchmark =====\n";

//     // Large config to stress memory
//     int batch = 4;
//     int seq_len = 512;
//     int embed_dim = 1024;
//     int num_heads = 16;
//     int num_layers = 12;
//     int ckpt_interval = 2;

//     std::cout << "Config: batch=" << batch
//               << " seq_len=" << seq_len
//               << " embed_dim=" << embed_dim
//               << " num_heads=" << num_heads
//               << " num_layers=" << num_layers
//               << " ckpt_interval=" << ckpt_interval << "\n\n";

//     std::cout << "[Run 1] Without checkpointing...\n";
//     auto no = run_transformer(batch, seq_len, embed_dim, num_heads,
//                               num_layers, ckpt_interval, false);

//     std::this_thread::sleep_for(std::chrono::milliseconds(500));

//     std::cout << "\n[Run 2] With checkpointing...\n";
//     auto ck = run_transformer(batch, seq_len, embed_dim, num_heads,
//                               num_layers, ckpt_interval, true);

//     // -------------------------------------------------------
//     // Report
//     // -------------------------------------------------------
//     std::cout << std::fixed << std::setprecision(2);
//     std::cout << "\n--- RESULTS ---\n";
//     std::cout << "Without checkpointing:\n";
//     std::cout << "  Forward activations: " << no.forward_mem_before << " MB\n";
//     std::cout << "  Backward time:       " << no.backward_time_ms << " ms\n\n";
//     std::cout << "With checkpointing:\n";
//     std::cout << "  Forward activations (before eviction): " << ck.forward_mem_before << " MB\n";
//     std::cout << "  After eviction:                       " << ck.forward_mem_after << " MB\n";
//     std::cout << "  Backward time (with recompute):       " << ck.backward_time_ms << " ms\n\n";

//     double mem_saved = 100.0 * (1.0 - ck.forward_mem_after / no.forward_mem_before);
//     double time_over = 100.0 * (ck.backward_time_ms / no.backward_time_ms - 1.0);
//     std::cout << "Memory saved: " << mem_saved << " %\n";
//     std::cout << "Recompute overhead: " << time_over << " %\n";

//     std::cout << "\nInterpretation:\n";
//     std::cout << "  • Expect memory_after < memory_before for checkpointed run.\n";
//     std::cout << "  • Expect backward_time_with_ckpt > without_ckpt.\n";
//     std::cout << "  • Smaller checkpoint_interval → more memory saved but higher compute cost.\n";
//     std::cout << "============================================\n";
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
#include <set>
using namespace ag;
using namespace checkpoint_impl;
// -----------------------------------------------------------
// Memory estimation helper
// -----------------------------------------------------------

// ---------------- Activation stats helper ----------------
struct ActivationStats {
    size_t total_nodes = 0;
    size_t stored_values = 0;      // nodes that currently have value
    size_t checkpointed = 0;       // nodes marked is_checkpoint
    size_t saved_snapshots = 0;    // total number of saved_input_tensors entries (non-empty)
    size_t saved_input_refs = 0;   // total number of saved_inputs that reference nodes
    size_t value_bytes = 0;        // bytes used by node->value
    size_t snapshot_bytes = 0;     // bytes used by saved_input_tensors copies
    std::vector<std::string> samples_with_value; // short list of node debug names with values
    std::vector<std::string> samples_cleared;    // list of nodes cleared after eviction (filled by diff)
};

// Walk graph and compute activation/snapshot stats
ActivationStats analyze_graph_activations(const ag::Value &root, size_t sample_limit = 8) {
    ActivationStats s;
    if (!root.node) return s;
    std::unordered_set<Node*> seen;
    std::deque<std::shared_ptr<Node>> q;
    q.push_back(root.node);

    while (!q.empty()) {
        auto n = q.front(); q.pop_front();
        if (!n || seen.count(n.get())) continue;
        seen.insert(n.get());
        ++s.total_nodes;

        // stored forward value?
        if (n->value.numel() != 0) {
            ++s.stored_values;
            s.value_bytes += n->value.numel() * sizeof(float);
            if (s.samples_with_value.size() < sample_limit) {
                std::ostringstream ss;
                ss << (n->debug_name ? n->debug_name : "(null)") << "@" << n.get()
                   << " [" << n->value.rows() << "x" << n->value.cols() << "]";
                s.samples_with_value.push_back(ss.str());
            }
        }

        if (n->is_checkpoint) ++s.checkpointed;

        // saved_input_tensors snapshots (explicit copies taken when capturing)
        for (size_t i = 0; i < n->saved_input_tensors.size(); ++i) {
            const Tensor &t = n->saved_input_tensors[i];
            if (t.numel() != 0) {
                ++s.saved_snapshots;
                s.snapshot_bytes += t.numel() * sizeof(float);
            }
        }

        // saved_inputs references to nodes
        for (size_t i = 0; i < n->saved_inputs.size(); ++i) {
            const Value &sv = n->saved_inputs[i];
            if (sv.node) ++s.saved_input_refs;
        }

        for (auto &p : n->inputs)
            if (p) q.push_back(p);
    }
    return s;
}

// Print a compact report
void print_activation_stats(const std::string &title, const ActivationStats &s) {
    std::cout << "---- " << title << " ----\n";
    std::cout << " total nodes:          " << s.total_nodes << "\n";
    std::cout << " stored forward vals:  " << s.stored_values << "  (" << std::fixed << std::setprecision(3)
              << (double)s.value_bytes / (1024.0*1024.0) << " MB)\n";
    std::cout << " checkpointed nodes:   " << s.checkpointed << "\n";
    std::cout << " saved snapshots:      " << s.saved_snapshots << "  (" << std::fixed << std::setprecision(3)
              << (double)s.snapshot_bytes / (1024.0*1024.0) << " MB)\n";
    std::cout << " saved input refs:     " << s.saved_input_refs << "\n";
    // if (!s.samples_with_value.empty()) {
    //     // std::cout << " sample nodes with value (up to 8):\n";
    //     for (auto &x : s.samples_with_value) std::cout << "   - " << x << "\n";
    // }
    std::cout << "---------------------------\n";
}

// Call existing evict_non_checkpoint_values but produce a before/after diff report
ActivationStats evict_and_report(const ag::Value &root, size_t show_diff_limit = 12) {
    // Snapshot before
    ActivationStats before = analyze_graph_activations(root, 1000); // collect many names
    // Build set of nodes that had values before
    std::unordered_set<Node*> had_value;
    if (root.node) {
        std::deque<std::shared_ptr<Node>> q;
        std::unordered_set<Node*> seen;
        q.push_back(root.node);
        while (!q.empty()) {
            auto n = q.front(); q.pop_front();
            if (!n || seen.count(n.get())) continue;
            seen.insert(n.get());
            if (n->value.numel() != 0) had_value.insert(n.get());
            for (auto &p : n->inputs) if (p) q.push_back(p);
        }
    }

    // Perform eviction using your existing function
    evict_non_checkpoint_values(root);

    // Snapshot after
    ActivationStats after = analyze_graph_activations(root, 1000);

    // Build list of nodes cleared
    std::vector<std::string> cleared_list;
    if (root.node) {
        std::deque<std::shared_ptr<Node>> q;
        std::unordered_set<Node*> seen;
        q.push_back(root.node);
        while (!q.empty()) {
            auto n = q.front(); q.pop_front();
            if (!n || seen.count(n.get())) continue;
            seen.insert(n.get());
            bool before_has = had_value.count(n.get());
            bool after_has = (n->value.numel() != 0);
            if (before_has && !after_has) {
                std::ostringstream ss;
                ss << (n->debug_name ? n->debug_name : "(null)") << "@" << n.get()
                   << " was cleared (val_size before unknown)";
                cleared_list.push_back(ss.str());
            }
            for (auto &p : n->inputs) if (p) q.push_back(p);
        }
    }

    // Print report: before / after
    print_activation_stats("BEFORE EVICT", before);
    print_activation_stats("AFTER  EVICT", after);

    size_t show_n = std::min(show_diff_limit, cleared_list.size());
    if (!cleared_list.empty()) {
        std::cout << "Nodes cleared by eviction (showing up to " << show_n << "):\n";
        for (size_t i = 0; i < show_n; ++i) std::cout << "  - " << cleared_list[i] << "\n";
        if (cleared_list.size() > show_n) std::cout << "  ... and " << (cleared_list.size()-show_n) << " more\n";
    } else {
        std::cout << "No nodes were cleared by eviction.\n";
    }

    // return the after stats for caller if needed
    return after;
}

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
    compute_forward_values(loss);
    capture_checkpoint_snapshots(loss);
    // 4. Evict activations for checkpoint version
    GraphMem m_after;
    if (use_ckpt) {
        ActivationStats after_stats = evict_and_report(loss);
        m_after.value_bytes = after_stats.value_bytes;
        m_after.grad_bytes = 0;  // optional
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
// RunResult run_network(int batch, int input_dim, int hidden, int depth,
//                       int checkpoint_interval, bool use_ckpt)
// {
//     // 1. Build data + params
//     Tensor x_data = Tensor::randn(batch, input_dim, 123);
//     Value x = constant(x_data, "x");

//     std::vector<Value> Wq, Wk, Wv, W1, W2, b1, b2;
//     for (int l = 0; l < depth; ++l) {
//         // 3× linear projections per layer (like attention)
//         Wq.push_back(param(Tensor::randn((l == 0) ? input_dim : hidden, hidden, 100 + l), ("Wq" + std::to_string(l)).c_str()));
//         Wk.push_back(param(Tensor::randn((l == 0) ? input_dim : hidden, hidden, 200 + l), ("Wk" + std::to_string(l)).c_str()));
//         Wv.push_back(param(Tensor::randn((l == 0) ? input_dim : hidden, hidden, 300 + l), ("Wv" + std::to_string(l)).c_str()));

//         // Feed-forward MLP inside each block
//         W1.push_back(param(Tensor::randn(hidden, hidden * 4, 400 + l), ("W1" + std::to_string(l)).c_str()));
//         b1.push_back(param(Tensor::randn(1, hidden * 4, 500 + l), ("b1" + std::to_string(l)).c_str()));
//         W2.push_back(param(Tensor::randn(hidden * 4, hidden, 600 + l), ("W2" + std::to_string(l)).c_str()));
//         b2.push_back(param(Tensor::randn(1, hidden, 700 + l), ("b2" + std::to_string(l)).c_str()));
//     }

//     Tensor Wout_t = Tensor::randn(hidden, input_dim, 800);
//     Tensor bout_t = Tensor::randn(1, input_dim, 900);
//     Value Wout = param(Wout_t, "Wout");
//     Value bout = param(bout_t, "bout");

//     // 2. Forward graph
//     Value cur = x;
//     for (int l = 0; l < depth; ++l) {
//         // Layer input normalization
//         cur = rms(cur);

//         // "Attention-like" projections (heavy matmuls)
//         Value q = matmul(cur, Wq[l]);
//         Value k = matmul(cur, Wk[l]);
//         Value v = matmul(cur, Wv[l]);

//         // Pairwise interaction (dot-product + softmax)
//         Value logits = matmul(q, transpose(k));
//         Value weights = softmax_row(logits);
//         Value attn = matmul(weights, v);

//         // Skip connection
//         Value skip1 = add(cur, attn);

//         // Optional checkpoint every few layers
//         // if (use_ckpt && ((l + 1) % checkpoint_interval) == 0)
//             skip1 = checkpoint(skip1, CheckpointOptions());

//         // Feed-forward (two dense + activation)
//         Value ff = add(matmul(skip1, W1[l]), b1[l]);
//         ff = gelu(ff);
//         ff = add(matmul(ff, W2[l]), b2[l]);

//         // Second skip + nonlinearity
//         cur = relu(add(skip1, ff));
//     }

//     // Output projection + reduction
//     cur = add(matmul(cur, Wout), bout);
//     Value loss = sum(cur);  // scalar loss

//     // 3. Measure forward activation memory
//     GraphMem m_before = estimate_graph_memory(loss);

//     // 4. Evict activations if using checkpointing
//     GraphMem m_after;
//     if (use_ckpt) {
//     ActivationStats after_stats = evict_and_report(loss);
//     m_after.value_bytes = after_stats.value_bytes;
//     m_after.grad_bytes = 0;  // optional
//     } else {
//         m_after = m_before;
//     }
//     // 5. Backward timing (includes recomputation)
//     auto t0 = std::chrono::high_resolution_clock::now();
//     backward(loss);
//     auto t1 = std::chrono::high_resolution_clock::now();
//     double t_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

//     return {toMB(m_before.value_bytes), toMB(m_after.value_bytes), t_ms};
// }

// RunResult run_network(int batch, int input_dim, int hidden, int depth,
//                       int checkpoint_interval, bool use_ckpt)
// {
//     // 1. Input data
//     Tensor x_data = Tensor::randn(batch, input_dim, 123);
//     Value x = constant(x_data, "x");

//     // Input projection to match hidden dim
//     Tensor Win_t = Tensor::randn(input_dim, hidden, 42);
//     Tensor bin_t = Tensor::randn(1, hidden, 43);
//     Value Win = param(Win_t, "Win");
//     Value bin = param(bin_t, "bin");

//     Value cur = relu(add(matmul(x, Win), bin));  // (batch, hidden)

//     // 2. Parameters for all blocks
//     std::vector<Value> W1, b1, W2, b2;
//     for (int l = 0; l < depth; ++l) {
//         Tensor W1_t = Tensor::randn(hidden, hidden * 4, 100 + l);
//         Tensor b1_t = Tensor::randn(1, hidden * 4, 200 + l);
//         Tensor W2_t = Tensor::randn(hidden * 4, hidden, 300 + l);
//         Tensor b2_t = Tensor::randn(1, hidden, 400 + l);
//         W1.push_back(param(W1_t, ("W1_" + std::to_string(l)).c_str()));
//         b1.push_back(param(b1_t, ("b1_" + std::to_string(l)).c_str()));
//         W2.push_back(param(W2_t, ("W2_" + std::to_string(l)).c_str()));
//         b2.push_back(param(b2_t, ("b2_" + std::to_string(l)).c_str()));
//     }

//     Tensor Wout_t = Tensor::randn(hidden, input_dim, 500);
//     Tensor bout_t = Tensor::randn(1, input_dim, 600);
//     Value Wout = param(Wout_t, "Wout");
//     Value bout = param(bout_t, "bout");

//     // 3. Forward graph — complex MLP + checkpointing
//     for (int l = 0; l < depth; ++l) {
//         // normalize input
//         cur = rms(cur);

//         // heavy feed-forward
//         Value z1 = add(matmul(cur, W1[l]), b1[l]);
//         Value z2 = gelu(z1);

//         // optional checkpoint between sub-layers
//         if (use_ckpt && ((l + 1) % checkpoint_interval) == 0)
//             z2 = checkpoint(z2, CheckpointOptions());

//         Value z3 = rms(z2);
//         Value z4 = add(matmul(z3, W2[l]), b2[l]);

//         // residual + activation
//         cur = relu(add(cur, z4));
//     }

//     // 4. Output head
//     cur = add(matmul(cur, Wout), bout);
//     Value loss = sum(cur);  // scalar

//     // 5. Memory measurement
//     GraphMem m_before = estimate_graph_memory(loss);
    
//   GraphMem m_after;
//     if (use_ckpt) {
//         ActivationStats after_stats = evict_and_report(loss);
//         m_after.value_bytes = after_stats.value_bytes;
//         m_after.grad_bytes = 0;  // optional
//     } else {
//         m_after = m_before;
//     }

//     // 6. Backward + timing
//     auto t0 = std::chrono::high_resolution_clock::now();
//     backward(loss);
//     auto t1 = std::chrono::high_resolution_clock::now();
//     double t_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

//     return {toMB(m_before.value_bytes), toMB(m_after.value_bytes), t_ms};
// }


// -----------------------------------------------------------
// Benchmark runner
// -----------------------------------------------------------
int main(){
    std::cout << "===== Gradient Checkpointing: Memory vs Time =====\n";
    // Large enough network to show a clear difference
    int batch = 8;
    int input_dim = 512;
    int hidden = 1024;
    int depth  = 128;
    int ckpt_interval = 2;

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
