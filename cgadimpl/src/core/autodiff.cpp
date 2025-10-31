// =============================================
// cgadimpl/src/core/autodiff.cpp
// =============================================
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include "ad/autodiff.hpp"
#include "ad/detail/autodiff_ops.hpp"
#include "ad/debug.hpp"
#include <ad/checkpoint.hpp>
namespace ag {

void zero_grad(const Value& root){
    auto order = topo_from(root.node.get());
    for (Node* n : order) if (n->requires_grad) n->grad = Tensor::zeros_like(n->value);
}

// void backward(const Value& root, const Tensor* grad_seed = nullptr){
//     if (!root.node) return;
//     auto order = topo_from(root.node.get());

//     // seed
//     if (root.node->requires_grad) {
//         root.node->grad = grad_seed ? *grad_seed
//                                     : (root.node->value.size()==1 ? Tensor::ones(1,1)
//                                                                   : Tensor::ones_like(root.node->value));
//     }

//     // reverse topo
//     for (auto it = order.rbegin(); it != order.rend(); ++it) {
//         Node* n = *it;
//         if (!n->requires_grad) continue;
//         const Tensor& gy = n->grad;

//         ag::debug::on_backprop_step(n, gy); // (optional) prints one line per node

//         if (n->is_checkpoint && n->value.size() == 0) {
//         if (!ag::checkpoint_impl::recompute_subgraph(n->shared_from_this())) {
//             throw std::runtime_error("autodiff: failed to recompute checkpointed node during backward");
//         }
//         }
//         VjpFn fn = vjp_lookup(n->op);
//         if (fn) fn(n, gy); // handler accumulates into parents
//     }
// }

void backward(const Value& root, const Tensor* grad_seed) {
    if (!root.node) return;

    auto order = topo_from(root.node.get());  // topological order (parents before child)

    // 1️⃣ Seed gradient at the root node
    if (root.node->requires_grad) {
        root.node->grad = grad_seed ? *grad_seed
                                    : (root.node->value.size() == 1
                                           ? Tensor::ones(1, 1)
                                           : Tensor::ones_like(root.node->value));
    }

    // 2️⃣ Iterate in reverse topological order (child → parent)
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
        if (!n || !n->requires_grad) continue;

        const Tensor& gy = n->grad;

        // optional debug: print step info
        ag::debug::on_backprop_step(n, gy);

        // If this node was checkpointed and has no value, recompute it
        if (n->is_checkpoint && n->value.size() == 0) {
            if (!ag::checkpoint_impl::recompute_subgraph(n->shared_from_this())) {
                std::ostringstream ss;
                ss << "[backward] ERROR: failed to recompute checkpointed node "
                   << n << " (\"" << (n->debug_name ? n->debug_name : "(null)") << "\")";
                throw std::runtime_error(ss.str());
            }
        }

        // 3️⃣ Ensure parent nodes have valid values before applying VJP
        for (auto& p_sp : n->inputs) {
            if (!p_sp) continue;

            // If parent's value is missing, handle appropriately
            if (p_sp->value.numel() == 0) {
                if (p_sp->is_checkpoint) {
                    // Try recomputing the missing checkpointed parent
                    if (!ag::checkpoint_impl::recompute_subgraph(p_sp)) {
                        std::ostringstream ss;
                        ss << "[backward] ERROR: failed to recompute parent checkpoint "
                           << p_sp.get() << " (\"" << (p_sp->debug_name ? p_sp->debug_name : "(null)") << "\")"
                           << " required by node " << n
                           << " (\"" << (n->debug_name ? n->debug_name : "(null)") << "\")";
                        throw std::runtime_error(ss.str());
                    }
                } else {
                    // Parent isn't checkpointed → this is a real error
                    std::ostringstream ss;
                    ss << "[backward] ERROR: parent value empty but not checkpointed: "
                       << (p_sp->debug_name ? p_sp->debug_name : "(null)")
                       << " (node=" << p_sp.get() << ")"
                       << " required by " << (n->debug_name ? n->debug_name : "(null)")
                       << " (node=" << n << ")";
                    std::cerr << ss.str() << "\n";
                    throw std::runtime_error(ss.str());
                }
            }
        }

        // 4️⃣ Get VJP function (gradient rule)
        VjpFn fn = vjp_lookup(n->op);
        if (!fn) {
            std::cerr << "[backward] WARNING: no VJP registered for op="
                      << static_cast<int>(n->op)
                      << " (" << (n->debug_name ? n->debug_name : "(null)") << ")\n";
            continue;
        }

        // 5️⃣ Apply vector-Jacobian product (accumulates grads into parents)
        try {
            fn(n, gy);
        } catch (const std::exception& e) {
            std::ostringstream ss;
            ss << "[backward] Exception in VJP for node "
               << n << " (" << (n->debug_name ? n->debug_name : "(null)") << "): "
               << e.what();
            throw std::runtime_error(ss.str());
        }
    }
}


Tensor jvp(const Value& root, const std::unordered_map<Node*, Tensor>& seed){
    if (!root.node) return Tensor{};
    auto order = topo_from(root.node.get());
    std::unordered_map<Node*, Tensor> T;
    T.reserve(order.size());

    auto tangent_of = [&](Node* p) -> const Tensor& {
        auto it = T.find(p);
        if (it != T.end()) return it->second;
        static Tensor Z; // fallback; shouldn't be used for leaves without seeds
        return Z;
    };

    for (Node* n : order) {
        // seed tangent for this node (if provided), else zeros
        Tensor t = Tensor::zeros_like(n->value);
        if (auto it = seed.find(n); it != seed.end()) t = it->second;

        ag::debug::on_jvp_step(n); // (optional) prints forward-mode step

        JvpFn fn = jvp_lookup(n->op);
        if (fn) t = fn(n, tangent_of);

        T[n] = t;
    }
    return T[root.node.get()];
}

} // namespace ag
