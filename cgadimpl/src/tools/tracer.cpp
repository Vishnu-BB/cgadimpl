// ad/tracer.cpp
#include "ad/tracer.hpp"
#include "ad/debug.hpp"

#include <stack>
#include <algorithm>
#include <unordered_set>
#include <iostream>

namespace ag::trace {

std::shared_ptr<Tracer> make_tracer() {
    return std::make_shared<Tracer>();
}

Tracer::~Tracer() {
    // Ensure we uninstall callback if still installed
    try { stop(); } catch(...) {}
}

void Tracer::start() {
    // install callback that forwards to this tracer instance
    ag::debug::set_node_created_callback(
        [this](const std::shared_ptr<Node>& n) {
            this->on_node_created(n);
        });
}


void Tracer::stop() {
    // remove the top-most callback we installed (thread-local stack semantics in debug.cpp)
    ag::debug::clear_node_created_callback();
}

void Tracer::on_node_created(const NodePtr& n) {
    if (!n) return;
    add_if_new(n);
}

void Tracer::add_if_new(const NodePtr& n) {
    if (!n) return;
    std::lock_guard<std::mutex> lk(mu_);
    Node* raw = n.get();
    if (seen_raw_.insert(raw).second) {
        order_.push_back(n);
    }
}

void Tracer::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    order_.clear();
    seen_raw_.clear();
    outputs_raw_.clear();
}

std::vector<NodePtr> Tracer::captured_nodes() const {
    std::lock_guard<std::mutex> lk(mu_);
    return order_;
}

void Tracer::mark_output(const NodePtr& n) {
    if (!n) return;
    std::lock_guard<std::mutex> lk(mu_);
    outputs_raw_.insert(n.get());
}

std::vector<NodePtr> Tracer::outputs() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<NodePtr> outs;
    if (!outputs_raw_.empty()) {
        // If explicit outputs exist, return matching shared_ptrs from order_
        for (auto &sp : order_) {
            if (outputs_raw_.count(sp.get())) outs.push_back(sp);
        }
        return outs;
    }
    // else detect: any captured node that is not an input to any other captured node
    std::unordered_set<Node*> has_consumer;
    for (auto &sp : order_) {
        // node->inputs likely is vector<NodePtr> or similar.
        for (auto &in : sp->inputs) {
            if (in) has_consumer.insert(in.get());
        }
    }
    for (auto &sp : order_) {
        if (has_consumer.count(sp.get()) == 0) outs.push_back(sp);
    }
    // fallback: if no nodes have zero consumers (e.g., trivial), mark last captured as output
    if (outs.empty() && !order_.empty()) outs.push_back(order_.back());
    return outs;
}

// Topological sort helper: DFS on captured subgraph.
// We'll build a set of raw pointers for fast membership check.
static void dfs_visit(Node* n,
                      const std::unordered_set<Node*>& captured_set,
                      std::unordered_set<Node*>& temp,
                      std::unordered_set<Node*>& perm,
                      std::vector<Node*>& out)
{
    if (!n) return;
    if (perm.count(n)) return;
    if (temp.count(n)) return; // cycle shouldn't occur; ignore
    temp.insert(n);
    // iterate inputs (Node likely keeps shared_ptrs in node->inputs)
    for (auto &in_sp : n->inputs) {
        if (!in_sp) continue;
        Node* in_raw = in_sp.get();
        if (captured_set.count(in_raw)) {
            dfs_visit(in_raw, captured_set, temp, perm, out);
        }
    }
    perm.insert(n);
    out.push_back(n);
}

std::vector<NodePtr> Tracer::topo_sort() const {
    std::vector<NodePtr> res;

    // Make a snapshot of captured nodes safely
    std::vector<NodePtr> order_copy;
    {
        std::lock_guard<std::mutex> lk(mu_);
        order_copy = order_;
    }

    if (order_copy.empty()) return res;

    // Build raw pointer set
    std::unordered_set<Node*> captured_set;
    captured_set.reserve(order_copy.size() * 2);
    for (auto &sp : order_copy) captured_set.insert(sp.get());

    // Now safely call outputs() — it will take its own lock
    std::vector<NodePtr> outs = outputs();

    std::unordered_set<Node*> temp, perm;
    std::vector<Node*> out_raw;

    // DFS from outputs
    for (auto &out_sp : outs)
        dfs_visit(out_sp.get(), captured_set, temp, perm, out_raw);

    for (auto &sp : order_copy)
        if (!perm.count(sp.get()))
            dfs_visit(sp.get(), captured_set, temp, perm, out_raw);

    std::reverse(out_raw.begin(), out_raw.end());

    std::unordered_map<Node*, NodePtr> raw_to_sp;
    raw_to_sp.reserve(order_copy.size() * 2);
    for (auto &sp : order_copy) raw_to_sp[sp.get()] = sp;

    for (auto *r : out_raw)
        if (auto it = raw_to_sp.find(r); it != raw_to_sp.end())
            res.push_back(it->second);

    return res;
}


/* CaptureGuard implementation */
CaptureGuard::CaptureGuard(std::shared_ptr<Tracer> t) : tracer_(std::move(t)) {
    if (tracer_) tracer_->start();
}
CaptureGuard::~CaptureGuard() {
    if (tracer_) tracer_->stop();
}

} // namespace ag::trace
