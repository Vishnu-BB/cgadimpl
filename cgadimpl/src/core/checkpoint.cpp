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
    for (auto &p : node->inputs) {
        if (p)
            node->saved_inputs.emplace_back(Value(p));  // Save reference to parent node
        else
            node->saved_inputs.emplace_back(Value());   // Empty placeholder
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
    // for (size_t i = 0; i < node->saved_inputs.size() && i < node->inputs.size(); ++i) {
    //     const Value &sv = node->saved_inputs[i];
    //     auto parent = node->inputs[i];
    //     if (!parent) continue;

    //     if (sv.node) {
    //         // Directly restore parent tensor from saved Value
    //         parent->value = sv.node->value;
    //     } else {
    //         // If we don't have a saved value, but parent value is missing, try to recompute
    //         if (parent->value.size() == 0) {
    //             if (parent->is_checkpoint) {
    //                 // Recursive recomputation of parent checkpoint
    //                 if (!recompute_subgraph(parent)) {
    //                     std::cerr << "[checkpoint] failed to recompute parent\n";
    //                     return false;
    //                 }
    //             } else {
    //                 std::cerr << "[checkpoint] missing parent value and parent isn't checkpointed\n";
    //                 return false;
    //             }
    //         }
    //     }
    // }
std::cerr << "[recompute] node=" << node.get()
          << " name=" << node->debug_name
          << " inputs=" << node->inputs.size() << "\n";
for (size_t i = 0; i < node->saved_inputs.size() && i < node->inputs.size(); ++i) {
    const Value &sv = node->saved_inputs[i];
    auto parent = node->inputs[i];
    if (!parent) continue;
    auto s = parent->value.shape();
    std::cerr << "   input[" << i << "] "
              << (parent->is_checkpoint ? "[CKPT]" : "[FREE]")
              << " shape=" << s.first << "x" << s.second
              << " numel=" << parent->value.numel() << "\n";
    // 1) If we stored a snapshot tensor, restore it
    if (i < node->saved_input_tensors.size() && node->saved_input_tensors[i].size() != 0) {
        parent->value = node->saved_input_tensors[i];
        continue;
    }

    // 2) Otherwise, if the saved Value points to a node whose value exists, copy that
    if (sv.node && sv.node->value.size() != 0) {
        parent->value = sv.node->value;
        continue;
    }

    // 3) Else, if parent is a checkpoint, recurse; otherwise fail
    if (parent->value.size() == 0) {
        if (parent->is_checkpoint) {
            if (!recompute_subgraph(parent)) {
                std::cerr << "[checkpoint] failed to recompute parent\n";
                return false;
            }
        } else {
            std::cerr << "[checkpoint] missing parent value and parent isn't checkpointed\n";
            return false;
        }
    }
    for (auto &p : node->inputs) {
    if (p && p->value.size() == 0) {
        std::cerr << "[ERROR] recompute_subgraph: parent@" << p.get()
                  << " (" << (p->debug_name ? p->debug_name : "(null)")
                  << ") has empty tensor before recompute of node@" << node.get() << "\n";
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
void evict_non_checkpoint_values(const Value &root) {
    
    if (!root.node) return;
    std::unordered_set<Node*> seen;
    std::deque<std::shared_ptr<Node>> q;
    q.push_back(root.node);
    while (!q.empty()) {
        auto n = q.front(); q.pop_front();
        if (!n || seen.count(n.get())) continue;
        seen.insert(n.get());
        if (n->is_checkpoint) {
            n->value = Tensor();   // free memory
            n->tape.clear();       // clear intermediates
        }
        for (auto &p : n->inputs)
            if (p) q.push_back(p);
    }
}
// Ensure forward values exist for all nodes reachable from root (parents before child)
void compute_forward_values(const ag::Value &root) {
    if (!root.node) return;
    auto order = ag::topo_from(root.node.get()); // parents before child
    for (ag::Node* n : order) {
        if (!n) continue;
        if (n->value.size() == 0) {
            try {
                // adapter that accepts Node* exists in your code
                n->value = forward_eval_node(n);
            } catch (const std::exception &e) {
                std::cerr << "[compute_forward_values] forward eval error node=" << n
                          << " name=\"" << (n->debug_name ? n->debug_name : "(null)")
                          << "\" : " << e.what() << "\n";
            }
        }
    }
}

// After forward pass, capture parent tensor snapshots for checkpoint nodes.
// Only snapshot parents that are non-leaf and not themselves checkpointed
// (these are the ones that will be evicted).
void capture_checkpoint_snapshots(const ag::Value &root) {

    if (!root.node) return;
    auto order = ag::topo_from(root.node.get()); // parents before child
    for (ag::Node* n : order) {
        if (!n) continue;
        if (!n->is_checkpoint) continue;
        n->saved_input_tensors.clear();
        n->saved_input_tensors.reserve(n->inputs.size());
        for (auto &p_sp : n->inputs) {
            if (!p_sp) { n->saved_input_tensors.emplace_back(Tensor()); continue; }
            ag::Node* p = p_sp.get();
            // If parent has a value and is not guaranteed to persist, snapshot it
            if (p->value.size() != 0 && !(p->op == ag::Op::Leaf || p->is_checkpoint)) {
                n->saved_input_tensors.emplace_back(p->value); // copy
            } else {
                n->saved_input_tensors.emplace_back(Tensor()); // placeholder
            }
        }
        // Debug (optional):
        // size_t bytes = 0; for (auto &t : n->saved_input_tensors) bytes += t.numel()*sizeof(float);
        // std::cerr << "[capture] node="<<n<<" snap_count="<<n->saved_input_tensors.size()<<" bytes="<<bytes<<"\n";
    }
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
