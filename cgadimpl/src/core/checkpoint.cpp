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

    // std::cerr << "[checkpoint] saved_inputs_count=" << node->saved_inputs.size()
    //           << " for node=" << node.get() << "\n";

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
    using std::endl;
    if (!node) return false;
    if (!node->is_checkpoint) {
        std::cerr << "[checkpoint] recompute_subgraph called on non-checkpoint node="
                  << node.get() << " name=\"" << (node->debug_name?node->debug_name:"(null)") << "\"\n";
        return false;
    }

    // Basic sanity: we must have saved_inputs or snapshots for this checkpoint
    if (node->saved_inputs.empty() && node->saved_input_tensors.empty()) {
        std::cerr << "[checkpoint] no saved inputs nor snapshots for recompute -- node="
                  << node.get() << " name=\"" << (node->debug_name?node->debug_name:"(null)") << "\"\n";
        return false;
    }

    // Restore RNG if necessary
    if (node->has_saved_rng) restore_rng_state(node->saved_rng_blob);

    // std::cerr << "[recompute] START node=" << node.get()
    //           << " name=\"" << (node->debug_name?node->debug_name:"(null)") << "\""
    //           << " op=" << static_cast<int>(node->op)
    //           << " inputs=" << node->inputs.size()
    //           << " saved_inputs=" << node->saved_inputs.size()
    //           << " saved_snapshots=" << node->saved_input_tensors.size() << "\n";

    // 1) Restore each parent value from snapshot or saved Value (sv)
    for (size_t i = 0; i < node->inputs.size(); ++i) {
        auto parent = node->inputs[i];
        if (!parent) {
            std::cerr << "   [recompute] parent[" << i << "] is null\n";
            continue;
        }

        // If we have a raw snapshot tensor for this input, restore it
        if (i < node->saved_input_tensors.size() && node->saved_input_tensors[i].numel() != 0) {
            parent->value = node->saved_input_tensors[i];
            std::cerr << "   [recompute] restored parent["<<i<<"] from snapshot (numel="<<parent->value.numel()<<")\n";
            continue;
        }

        // If saved_inputs entry points to a node whose value exists, copy it.
        if (i < node->saved_inputs.size()) {
            const Value &sv = node->saved_inputs[i];
            if (sv.node && sv.node->value.numel() != 0) {
                parent->value = sv.node->value;
                // std::cerr << "   [recompute] copied parent["<<i<<"] value from saved node "<< sv.node.get() <<"\n";
                continue;
            }
        }

        // If parent value is missing but parent is checkpointed, recursively recompute it
        if (parent->value.numel() == 0) {
            if (parent->is_checkpoint) {
                // std::cerr << "   [recompute] parent["<<i<<"] is checkpoint, recursing into parent="<<parent.get()<<"\n";
                if (!recompute_subgraph(parent)) {
                    std::cerr << "[checkpoint] failed to recompute parent (idx="<<i<<") for node="<<node.get()<<"\n";
                    return false;
                }
                // after recursion, parent->value should be filled
                if (parent->value.numel() == 0) {
                    std::cerr << "[checkpoint] parent recompute returned but parent->value still empty parent="<<parent.get()<<"\n";
                    return false;
                }
            } else {
                std::cerr << "[checkpoint] missing parent value and parent isn't checkpointed: parent="<<parent.get()
                          << " name=\"" << (parent->debug_name?parent->debug_name:"(null)") << "\"\n";
                return false;
            }
        }

        // At this point parent->value should be non-empty
        // std::cerr << "   [recompute] parent["<<i<<"] final shape=" << parent->value.rows() << "x" << parent->value.cols()
        //           << " numel=" << parent->value.numel()
        //           << " checkpoint=" << parent->is_checkpoint << "\n";
    } // for inputs

    // Double-check: all inputs must now have values
    for (auto &p_sp : node->inputs) {
        if (!p_sp) continue;
        if (p_sp->value.numel() == 0) {
            // std::cerr << "[checkpoint] ERROR: input still empty before forward_eval_node for node="<<node.get()
            //           << " name=\"" << (node->debug_name?node->debug_name:"(null)") << "\""
            //           << " input_node="<<p_sp.get() << " name=\"" << (p_sp->debug_name? p_sp->debug_name : "(null)") << "\"\n";
            return false;
        }
    }

    // 2) Run forward op to compute node->value
    try {
        Tensor out = forward_eval_node(node.get());   // run forward evaluator
        node->value = out;
        // std::cerr << "[recompute] SUCCESS for node=" << node.get()
        //           << " name=\"" << (node->debug_name?node->debug_name:"(null)") << "\""
        //           << " out.shape=" << node->value.rows() << "x" << node->value.cols()
        //           << " numel=" << node->value.numel() << "\n";

        // Notify in-place trackers / bookkeeping that node was recomputed
        ag::inplace::on_recomputed(node.get());
    } catch (const std::exception &e) {
        std::cerr << "[checkpoint] recompute exception: " << e.what()
                  << " -- node=" << node.get()
                  << " name=\"" << (node->debug_name?node->debug_name:"(null)") << "\"\n";
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
