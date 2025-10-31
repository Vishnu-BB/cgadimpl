//============================================================
// file: cgadimpl/src/core/checkpoint.cpp
//============================================================

#include "ad/checkpoint.hpp"
#include <unordered_set>
#include <stdexcept>
#include <iostream>
#include <deque>
#include <queue>
#include "ad/inplace.hpp"   // for on_recomputed() notifications
#include <iomanip>
namespace ag {
namespace checkpoint_impl {

/*
 *  ============================================================
 *  Core concepts:
 *  ============================================================
 *  This source file implements the **activation checkpointing logic**:
 *
 *  Checkpointing is a memory-saving technique used during training.
 *  Instead of storing every intermediate activation for backprop,
 *  we “checkpoint” specific nodes (activations), discard others, and
 *  later recompute them on-demand during the backward pass.
 *
 *  This file contains:
 *    - Mechanisms to mark nodes as checkpoints (`mark_node_checkpoint`)
 *    - Logic to recompute lost intermediate activations (`recompute_subgraph`)
 *    - Helper utilities for auto-checkpointing (`auto_checkpoint_every_n`, `auto_checkpoint_by_depth`)
 *
 *  When a node is checkpointed:
 *      We store minimal information (inputs + RNG state)
 *      We do not store all intermediate tensors
 *      During backward(), if the node’s value is missing, we recompute it
 */

using NodePtr = std::shared_ptr<Node>;  // Convenient alias for shared node references

// ------------------------------------------------------------
// RNG State Handling (stubs)
// ------------------------------------------------------------

/*
 *  RNG (Random Number Generator) state is saved to ensure that when
 *  the graph is recomputed later, any random operations (like dropout)
 *  produce the same deterministic results.
 *
 *  This design uses an **opaque byte blob (`RngBlob`)** to store the RNG state.
 *  This keeps checkpointing logic independent of the RNG implementation.
 *  Frameworks like PyTorch save the entire CPU/GPU RNG state snapshot.
 */
using RngBlob = std::vector<uint8_t>;

/*
 *  save_rng_state():
 *  ------------------
 *  A stub implementation that would capture RNG seed, counter, and internal buffers.
 *  For now, returns an empty blob. Extend this if you integrate custom RNG handling.
 */
static RngBlob save_rng_state() {
    RngBlob b;
    // TODO: capture RNG state (seed/counters) into blob
    return b;
}

/*
 *  restore_rng_state():
 *  ---------------------
 *  Restores the RNG state before recomputation. Currently a no-op stub.
 *  In a full implementation, it would restore random seeds and generator counters
 *  so that recomputation yields identical stochastic behavior.
 */
static void restore_rng_state(const RngBlob &b) {
    (void)b; // suppress unused warning
    // TODO: restore RNG state from blob
}

// ------------------------------------------------------------
// mark_node_checkpoint()
// ------------------------------------------------------------

/*
 *  mark_node_checkpoint():
 *  ------------------------
 *  Marks a node in the computational graph as a **checkpoint boundary**.
 *  This function performs:
 *      1. Sets `node->is_checkpoint = true`
 *      2. Saves minimal inputs into `node->saved_inputs`
 *      3. Optionally saves RNG state
 *
 *  Purpose:
 *  - Checkpoint boundaries define recomputation points.
 *  - Only inputs to the checkpoint are stored, not all intermediates.
 *
 *  Parameters:
 *      node : shared_ptr<Node>  → node to checkpoint
 *      opts : CheckpointOptions → flags controlling behavior
 */
void mark_node_checkpoint(const NodePtr &node, const CheckpointOptions &opts) {
    if (!node) return;             // Safety check
    if (node->is_checkpoint) return; // Avoid re-marking the same node (idempotent)

    node->is_checkpoint = true;   // Mark this node as checkpointed

    // Debug message (optional)
    std::cerr << "[checkpoint] mark_node_checkpoint: node=" << node.get()
              << " name=\"" << (node->debug_name ? node->debug_name : "(null)") << "\""
              << " inputs=" << node->inputs.size() << "\n";

    /*
     *  We now store the minimal input information.
     *  - Each parent `Node` pointer from `inputs` is wrapped as a `Value`.
     *  - These Values allow re-accessing the parent tensors when recomputing.
     */
    node->saved_inputs.clear();
    node->saved_input_tensors.clear();
    node->saved_input_tensors.reserve(node->inputs.size());
    for (auto &p : node->inputs) {
        if (p && p->value.size() != 0) {
            // Save a copy of the parent's tensor for recomputation
            node->saved_input_tensors.emplace_back(p->value);
            node->saved_inputs.emplace_back(); // Placeholder Value (no need to save Node ptr)
        } else {
            node->saved_input_tensors.emplace_back(); // Empty optional
            node->saved_inputs.emplace_back(Value());   // Empty Value
        }
    }

    std::cerr << "[checkpoint] saved_inputs_count=" << node->saved_inputs.size()
              << " for node=" << node.get() << "\n";

    /*
     *  Optionally capture the RNG state if requested in options.
     *  This ensures deterministic recomputation when random ops exist.
     */
    if (opts.save_rng) {
        node->saved_rng_blob = save_rng_state();
        node->has_saved_rng = true;
    } else {
        node->has_saved_rng = false;
    }
}
void compute_forward_values(const Value &root) {
    if (!root.node) return;

    auto order = ag::topo_from(root.node.get());  // parents → children

    for (Node* n : order) {
        if (!n) continue;

        // Skip if already computed
        if (n->value.size() != 0) continue;

        try {
            n->value = forward_eval_node(n);
            ag::inplace::on_recomputed(n);  // update version table for consistency
            std::cerr << "[compute_forward_values] recomputed node@"
                      << n << " name=\"" << (n->debug_name ? n->debug_name : "(null)")
                      << "\" op=" << static_cast<int>(n->op)
                      << " shape=" << n->value.rows() << "x" << n->value.cols()
                      << "\n";
        } catch (const std::exception &e) {
            std::cerr << "[compute_forward_values] failed node@" << n
                      << " (" << (n->debug_name ? n->debug_name : "(null)")
                      << "): " << e.what() << "\n";
        }
    }
}
void capture_checkpoint_snapshots(const Value &root) {
    if (!root.node) return;
    auto order = ag::topo_from(root.node.get());

    for (Node* n : order) {
        if (!n || !n->is_checkpoint) continue;

        n->saved_input_tensors.clear();
        n->saved_input_tensors.reserve(n->inputs.size());

        for (auto &p_sp : n->inputs) {
            if (!p_sp) {
                n->saved_input_tensors.emplace_back(Tensor());
                continue;
            }

            Node* p = p_sp.get();
            if (p->value.size() != 0) {
                // ✅ Always save a copy for checkpoint inputs
                n->saved_input_tensors.emplace_back(p->value);
            } else {
                n->saved_input_tensors.emplace_back(Tensor());
            }
        }

        std::cerr << "[capture] checkpoint@" << n
                  << " name=\"" << (n->debug_name ? n->debug_name : "(null)") << "\" "
                  << "saved_inputs=" << n->inputs.size() << "\n";
    }
}

// ------------------------------------------------------------
// recompute_subgraph()
// ------------------------------------------------------------

/*
 *  recompute_subgraph():
 *  ----------------------
 *  Recomputes the output of a checkpointed node by:
 *      1. Restoring input tensors from saved checkpoints
 *      2. Recursively recomputing missing parent nodes if necessary
 *      3. Invoking the operation's forward pass (`forward_eval_node`)
 *
 *  Returns:
 *      true  → successful recomputation
 *      false → failed due to missing parents or exceptions
 *
 *  This is typically invoked automatically during backward propagation
 *  when a checkpointed node’s value is required but has been freed.
 */
bool recompute_subgraph(const std::shared_ptr<Node>& node) {
    if (!node) return false;
    if (!node->is_checkpoint) return false;

    // Sanity check — make sure checkpoint data exists
    if (node->saved_inputs.empty()) {
        std::cerr << "[checkpoint] no saved inputs for recompute -- node=" << node.get()
                  << " name=\"" << (node->debug_name ? node->debug_name : "(null)") << "\""
                  << " inputs=" << node->inputs.size()
                  << " is_checkpoint=" << node->is_checkpoint << "\n";
        return false;
    }

    // Restore RNG state if previously saved
    if (node->has_saved_rng) {
        restore_rng_state(node->saved_rng_blob);
    }

    /*
     *  Step 1: Restore or recompute all parent inputs
     *  ------------------------------------------------
     *  For each saved input:
     *      - If `saved_inputs[i]` holds a node with tensor value, restore it.
     *      - Otherwise, if the parent's value is missing, attempt recursive recomputation.
     */
    for (size_t i = 0; i < node->saved_inputs.size() && i < node->inputs.size(); ++i) {
        const Value &sv = node->saved_inputs[i];
        auto parent = node->inputs[i];
        if (!parent){
            std::cerr << "[recompute debug] missing parent at index " << i << "\n";
            continue;
        } 
        std::cerr << "[recompute debug] input["<<i<<"] node="<<parent.get()
              << " name=\"" << (parent->debug_name?parent->debug_name:"(null)") << "\""
              << " numel=" << parent->value.numel()
              << " shape=" << parent->value.rows() << "x" << parent->value.cols()
              << " op=" << static_cast<int>(parent->op) << "\n";
    if (parent->value.numel() == 0) {
        std::cerr << "[recompute debug] WARNING: empty input before forward_eval_node -> will attempt recompute or fail\n";
    }
        if (sv.node) {
            // Directly restore parent tensor from saved Value
            parent->value = sv.node->value;
        } else {
            // If we don't have a saved value, but parent value is missing, try to recompute
            if (parent->value.size() == 0) {
                if (parent->is_checkpoint) {
                    // Recursive recomputation of parent checkpoint
                    if (!recompute_subgraph(parent)) {
                        std::cerr << "[checkpoint] failed to recompute parent\n";
                        return false;
                    }
                } else {
                    std::cerr << "[checkpoint] missing parent value and parent isn't checkpointed\n";
                    return false;
                }
            }
        }
    }

    /*
     *  Step 2: Run the forward operation for this node.
     *  -------------------------------------------------
     *  Using the restored input tensors, we now recompute
     *  the output tensor by calling the node’s operator implementation.
     */
    try {
        Tensor out = forward_eval_node(node.get());   // Re-run forward computation
        node->value = out;                            // Save result
        ag::inplace::on_recomputed(node.get());       // Optional callback (for debugging/versioning)
    } catch (const std::exception &e) {
        std::cerr << "[checkpoint] recompute exception: " << e.what() << "\n";
        return false;
    }

    return true;
}
// // Safe eviction: do not evict nodes that are referenced by saved_inputs of any node
void evict_non_checkpoint_values(const Value &root) {
    if (!root.node) return;

    // 1) Mark protected nodes: nodes on any path from root down to (but NOT through) checkpoint nodes.
    std::unordered_set<Node*> protected_set;
    std::deque<std::shared_ptr<Node>> q;
    q.push_back(root.node);

    while (!q.empty()) {
        auto n = q.front(); q.pop_front();
        if (!n || protected_set.count(n.get())) continue;

        // Mark as protected (we must keep this node->value until backward or recompute)
        protected_set.insert(n.get());

        // If node is a checkpoint, stop descending — its inputs can be evicted (or are snapshotted).
        if (n->is_checkpoint) {
            continue;
        }

        // Otherwise descend into inputs and protect them (they are needed by backward)
        for (auto &p : n->inputs)
            if (p) q.push_back(p);
    }

    // 2) Sweep: clear values for nodes that are NOT protected.
    std::unordered_set<Node*> seen;
    std::deque<std::shared_ptr<Node>> q2;
    q2.push_back(root.node);

    size_t cleared_count = 0;
    size_t cleared_bytes = 0;

    while (!q2.empty()) {
        auto n = q2.front(); q2.pop_front();
        if (!n || seen.count(n.get())) continue;
        seen.insert(n.get());

        // If node is not protected, evict its forward value and tape.
        if (!protected_set.count(n.get())) {
            // Keep snapshots / saved_inputs for checkpoint nodes (they're stored separately)
            if (n->value.numel() != 0) {
                cleared_bytes += n->value.numel() * sizeof(float);
            }
            n->value = Tensor();   // free memory
            n->tape.clear();
            ++cleared_count;
        }

        for (auto &p : n->inputs)
            if (p) q2.push_back(p);
    }

    std::cerr << "[evict] Freed " << cleared_count << " activations (~"
              << (double)cleared_bytes / (1024.0*1024.0) << " MB)\n";
}

// ------------------------------------------------------------
// ensure_value_present()
// ------------------------------------------------------------

/*
 *  ensure_value_present():
 *  ------------------------
 *  A small helper that guarantees that a node has a valid value.
 *  - If the node’s tensor is already computed → returns true.
 *  - If it’s empty but checkpointed → triggers recomputation.
 *  - Otherwise → returns false.
 */
inline bool ensure_value_present(const NodePtr &node) {
    if (!node) return false;
    if (node->value.size() != 0) return true;
    if (node->is_checkpoint) return recompute_subgraph(node);
    return false;
}

// ------------------------------------------------------------
// is_checkpointed()
// ------------------------------------------------------------

/*
 *  is_checkpointed():
 *  -------------------
 *  Returns whether a node has been checkpoint-marked.
 *  Used in multiple utilities and internal checks.
 */
inline bool is_checkpointed(const NodePtr &node) {
    return node && node->is_checkpoint;
}

} // namespace checkpoint_impl

// ------------------------------------------------------------
// auto_checkpoint_every_n()
// ------------------------------------------------------------

/*
 *  auto_checkpoint_every_n():
 *  ---------------------------
 *  Traverses the computation graph starting from `root`
 *  using a **Breadth-First Search (BFS)** and automatically
 *  checkpoints every Nth node.
 *
 *  Parameters:
 *      root : root Value (usually final output)
 *      n    : interval at which to checkpoint nodes (e.g., n=5 → every 5th node)
 *
 *  Implementation:
 *    - Keeps track of visited nodes to avoid reprocessing.
 *    - Uses a counter to checkpoint every Nth node.
 *    - Stores nodes in a queue for BFS traversal.
 */
void auto_checkpoint_every_n(const Value &root, int n) {
    if (n <= 0 || !root.node) return;

    std::unordered_set<Node*> visited;
    std::deque<std::shared_ptr<Node>> q;
    q.push_back(root.node);
    int counter = 0;

    while (!q.empty()) {
        auto cur = q.front();
        q.pop_front();

        if (!cur || visited.count(cur.get())) continue;
        visited.insert(cur.get());

        // Checkpoint every Nth node (except leaf nodes)
        ++counter;
        if (counter % n == 0 && !cur->inputs.empty()) {
            checkpoint_impl::mark_node_checkpoint(cur, CheckpointOptions());
        }

        // Enqueue parents for traversal
        for (auto &p : cur->inputs)
            if (p) q.push_back(p);
    }
}

// ------------------------------------------------------------
// auto_checkpoint_by_depth()
// ------------------------------------------------------------

/*
 *  auto_checkpoint_by_depth():
 *  ----------------------------
 *  Another BFS-based automatic checkpointing strategy.
 *  Instead of checkpointing every Nth node, it checkpoints
 *  nodes that exceed a **depth threshold** from the root.
 *
 *  Parameters:
 *      root : starting node
 *      depth_threshold : checkpoint nodes deeper than this value
 *
 *  This approach is often used in large sequential models
 *  (e.g., Transformers or RNNs) where deeper layers consume more memory.
 */
void auto_checkpoint_by_depth(const Value& root, int depth_threshold) {
    if (!root.node) return;

    struct QItem { std::shared_ptr<Node> node; int depth; };
    std::queue<QItem> q;
    std::unordered_set<Node*> visited;

    q.push({root.node, 0});
    while (!q.empty()) {
        auto [cur, depth] = q.front();
        q.pop();

        if (!cur || visited.count(cur.get())) continue;
        visited.insert(cur.get());

        // Mark nodes deeper than the given threshold
        if (depth >= depth_threshold && !cur->inputs.empty()) {
            checkpoint_impl::mark_node_checkpoint(cur, CheckpointOptions());
        }

        // Enqueue children with incremented depth
        for (auto &p : cur->inputs)
            if (p) q.push({p, depth + 1});
    }
}

} // namespace ag
