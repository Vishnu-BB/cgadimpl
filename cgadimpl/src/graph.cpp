// =====================
// file: src/graph.cpp
// =====================
#include <unordered_set>
#include <functional>
#include "ad/graph.hpp"


namespace ag {


    Node::Node() = default;
    Node::Node(const Tensor& v, bool rg, Op op_, const char* nm) 
        : op(op_), value(v), grad(Tensor::zeros_like(v)), requires_grad(rg), debug_name(nm) {}


    Value::Value() = default;

    Value::Value(std::shared_ptr<Node> n): node(std::move(n)) {}

    const Tensor& Value::val() const { 
        return node->value; 
    }

    Tensor& Value::grad() { 
        return node->grad; 
    }
    
    std::pair<int,int> Value::shape() const { 
        return node->value.shape(); 
    }


    Value constant(const Tensor& v, const char* name){ 
        return Value(std::make_shared<Node>(v,false,Op::Leaf,name)); 
    }

    Value param (const Tensor& v, const char* name){ 
        return Value(std::make_shared<Node>(v,true ,Op::Leaf,name));
    }

    // std::vector<Node*> topo_from(Node* root){
    //     std::vector<Node*> order; order.reserve(256);
    //     std::unordered_set<Node*> vis; vis.reserve(256);
    //     std::function<void(Node*)> dfs = [&](Node* n){ if(!n || vis.count(n)) return; vis.insert(n); for(auto& p : n->inputs) dfs(p.get()); order.push_back(n); };
    //     dfs(root);
    //     return order; // parents before child
    // }
    std::vector<Node*> topo_from(Node* root) {
    std::vector<Node*> order; order.reserve(256);
    std::unordered_set<Node*> vis; vis.reserve(256);
    std::function<void(Node*)> dfs = [&](Node* n) {
        if (!n || vis.count(n)) return;
        vis.insert(n);
        for (auto& p : n->inputs) {
            if (p) dfs(p.get());
        }
        order.push_back(n);
    };
    dfs(root);
    return order; // parents before child
}

void delete_subgraph(Node* root) {
    if (!root) return;

    // 1) Collect reachable nodes with a safe DFS
    std::unordered_set<Node*> visited;
    visited.reserve(256);
    std::function<void(Node*)> dfs = [&](Node* n) {
        if (!n || visited.count(n)) return;
        visited.insert(n);
        for (auto& inp_sp : n->inputs) {
            if (inp_sp) {
                Node* child = inp_sp.get();
                if (child && !visited.count(child)) dfs(child);
            }
        }
    };
    dfs(root);

    if (visited.empty()) return;

    // 2) For each visited node, release its input shared_ptrs safely,
    //    then clear the inputs vector. This breaks graph-owned references
    //    and allows nodes that have no other owners to be destroyed.
    //
    //    We never dereference a raw pointer obtained from a shared_ptr
    //    after we reset it, and we wrap resets in try/catch to be defensive.
    for (Node* n : visited) {
        if (!n) continue;
        for (auto& inp_sp : n->inputs) {
            if (!inp_sp) continue; // already null, skip
            try {
                // Resetting the shared_ptr reduces the reference count.
                // If use_count was 1, this destroys the pointed Node.
                // If other owners exist, resetting is still safe (it just
                // drops our reference).
                inp_sp.reset();
            } catch (...) {
                // Best-effort cleanup: swallow exceptions to avoid crashing.
            }
        }
        // Clear inputs vector to free capacity and make future calls safe.
        std::vector<std::shared_ptr<Node>> empty;
        n->inputs.swap(empty);
    }
}

} // namespace ag