#pragma once
// ad/tracer.hpp
// Lightweight tracer for capturing node creation during forward execution.
// Designed to integrate with existing ad::debug::on_node_created hook.

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <functional>
#include <mutex>

#include "ad/graph.hpp" // must contain definition of class Node and NodePtr (shared_ptr<Node>)

namespace ag::trace {

// Forward declaration
class Tracer;

// Type alias for convenience
using NodePtr = std::shared_ptr<Node>;

// Tracer collects Node shared_ptrs when nodes are created (callback from ag::debug).
// It keeps insertion order, deduplicates, can topo-sort, and can mark outputs.
class Tracer {
public:
    Tracer() = default;
    ~Tracer();

    // Start/stop capture. Safe to call multiple times (start/stop pairs).
    // These install/uninstall the callback into ag::debug.
    void start();
    void stop();

    // Called by the debug callback when a node is created.
    // Thread-safe.
    void on_node_created(const NodePtr& n);

    // Clear captured state.
    void clear();

    // Return captured nodes in capture insertion order (shared_ptrs).
    std::vector<NodePtr> captured_nodes() const;

    // Mark a node as an explicit output of the capture (optional).
    void mark_output(const NodePtr& n);

    // Return outputs explicitly marked; if none marked, detect outputs automatically:
    // nodes in captured set that are not used as inputs of any captured node.
    std::vector<NodePtr> outputs() const;

    // Topologically sort the subgraph consisting of captured nodes.
    // Returns nodes in parent-before-child order suitable for lowering/export.
    std::vector<NodePtr> topo_sort() const;

private:
    // Helper: add node only if not seen. Thread-safe.
    void add_if_new(const NodePtr& n);

    // Internal storage (protected by mutex).
    mutable std::mutex mu_;
    std::vector<NodePtr> order_;                 // insertion order
    std::unordered_set<Node*> seen_raw_;         // dedupe by raw pointer
    std::unordered_set<Node*> outputs_raw_;      // marked outputs
};

// RAII guard for simple usage:
struct CaptureGuard {
    explicit CaptureGuard(std::shared_ptr<Tracer> t);
    ~CaptureGuard();
private:
    std::shared_ptr<Tracer> tracer_;
};

std::shared_ptr<Tracer> make_tracer();

} // namespace ag::trace
