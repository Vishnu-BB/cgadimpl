// =====================
// file: src/ops.cpp
// =====================
#include "ad/ops.hpp"
#include "ad/nodeops.hpp" // Include the new node-level declarations
#include "ad/inplace.hpp"

#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace ag {

    Value add(const Value& a, const Value& b){ 
        return Value(detail::add_nodeops(a.node, b.node)); 
    }

    Value sub(const Value& a, const Value& b){ 
        
        return Value(detail::sub_nodeops(a.node, b.node)); 
    }
    Value inplace_checkpoint(const Value& v) {
        if (!v.node) return v;
        ag::inplace::mark_inplace_checkpoint(v.node);
        return v;
    }


    Value mul(const Value& a, const Value& b){ 
        return Value(detail::mul_nodeops(a.node, b.node)); 
    }

    Value flomul(const Value& a, float b){ 
        return Value(detail::flomul_nodeops(a.node, b));
    }

    Value relu(const Value& x){ 
      
        return Value(detail::relu_nodeops(x.node));
    }





    Value matmul(const Value& a, const Value& b){ 
         return Value(detail::matmul_nodeops(a.node, b.node)); 
    }

    Value fmab(const Value& a, const Value& b, const Value& c){ 
        return Value(detail::fmab_nodeops(a.node, b.node, c.node)); 
    }


    Value attention(const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(detail::attention_nodeops(a.node, b.node, c.node, d.node));
    }


    Value alibiatt(const Value& a, const Value& b, const Value& c, const Value& d, float m) { 
    return Value(detail::alibiatt_nodeops(a.node, b.node, c.node, d.node, m));
}



    Value swiglu(const Value& x, const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(detail::swiglu_nodeops(x.node, a.node, b.node, c.node, d.node));
    }


    Value sum(const Value& x){ 
        return Value(detail::sum_nodeops(x.node));
    }

    Value transpose(const Value& x){ 
        return Value(detail::transpose_nodeops(x.node));
    }

    Value exp(const Value& x){ 
        return Value(detail::exp_nodeops(x.node));
    }
    
    Value log(const Value& x){ 
        return Value(detail::exp_nodeops(x.node));
    }


    Value mish(const Value& x){ 
        return Value(detail::mish_nodeops(x.node));
    }
    
    Value tanh(const Value& x){ 
        return Value(detail::tanh_nodeops(x.node));
    }
    
    Value sigmoid(const Value& x){ 
        return Value(detail::sigmoid_nodeops(x.node));
    }
    
    Value softplus(const Value& x){ 
        return Value(detail::softplus_nodeops(x.node));
    }

    Value gaus(const Value& x){ 
        return Value(detail::gaus_nodeops(x.node));
    }
    
    Value gelu(const Value& x){ 
        return Value(detail::gelu_nodeops(x.node));
    }



    Value gcu(const Value& x){ 
        return Value(detail::gcu_nodeops(x.node));
    }
    
    Value silu(const Value& x){ 
        return Value(detail::silu_nodeops(x.node));
    }

    Value parcon(const Value& x){ 
        return Value(detail::parcon_nodeops(x.node));
    }

    Value lisht(const Value& x){ 
        return Value(detail::lisht_nodeops(x.node));
    }
    
    Value leaky_relu(const Value& x, float alpha){ 
        return Value(detail::leaky_relu_nodeops(x.node, alpha));
    }


    Value rowsum(const Value& x){ 
        return Value(detail::rowsum_nodeops(x.node));
    }
    
    Value rowmax(const Value& x){ 
        return Value(detail::rowmax_nodeops(x.node));
    }

    Value rms(const Value& x){ 
return Value(detail::rms_nodeops(x.node));
    }

    Value realrms(const Value& x, float g){ 
return Value(detail::realrms_nodeops(x.node, g));
    }

    Value laynor(const Value& x){ 
        return Value(detail::laynor_nodeops(x.node));
    }

    Value relaynor(const Value& x, float b, float g){ 
        return Value(detail::relaynor_nodeops(x.node, b, g));
    }
    
    Value mean_all(const Value& x){ 
        return Value(detail::mean_all_nodeops(x.node));
    }

    Value dyntanh(const Value& x, float a, float b, float g){ 
        return Value(detail::dyntanh_nodeops(x.node, a, b, g));
    }
    
    Value softmax_row(const Value& z){ 
        return Value(detail::softmax_row_nodeops(z.node));
    }
    
    Value logsumexp_row(const Value& z){ 
        return Value(detail::logsumexp_row_nodeops(z.node));
    }


    Value mambassm(const Value& z, const Value& a, const Value& b, const Value& c, const Value& d){ 

        return Value(detail::mambassm_nodeops(z.node, a.node, b.node, c.node, d.node));

        
    }


    Value cross_entropy_with_logits(const Value& logits, const Value& onehot){
    // Stable CE = mean( -sum(onehot * _nodeops(logits - logsumexp_row_nodeops(logits))) )
        return Value(detail::cross_entropy_with_logits_nodeops(logits.node, onehot.node));
    }


    Value kldivergence(const Value& logits, const Value& onehot){
        return Value(detail::kldivergence_nodeops(logits.node, onehot.node));
    }

    Value mse_loss(const Value& pred, const Value& target) {
    return Value(detail::mse_loss_nodeops(pred.node, target.node));
}


    Value mae_loss(const Value& pred, const Value& target) {
    return Value(detail::mae_loss_nodeops(pred.node, target.node));
}

//  The implementation of **forward evaluation logic** for a single
// computational graph node (`Node`) in the autodiff system.
//
// The purpose of `forward_eval_node()` is to *recompute* or *evaluate*
// a node’s output tensor based solely on its input nodes’ values,
// without using stored intermediate data.
//
// This is crucial for:
//    - Checkpoint recomputation (freeing and restoring activations),
//    - Lazy evaluation (on-demand computation),
//    - Debug visualization or forward-only inference.
//
// Additionally, the `checkpoint()` function here provides a user-facing API
// for marking specific nodes as checkpoints, integrating with the
// `checkpoint_impl` subsystem.
//
// Together, these functions enable **memory-efficient recomputation**
// during backward passes and safe graph traversal.
//


// -----------------------------------------------------------------------------
// forward_eval_node (shared_ptr<Node> version)
// -----------------------------------------------------------------------------

/*
 *  forward_eval_node():
 *  ---------------------
 *  Evaluates (or recomputes) the output tensor of a single computational node.
 *
 *  Parameters:
 *      - node : shared_ptr<Node> representing a node in the computational graph.
 *
 *  Returns:
 *      - A new Tensor that represents the computed output of this node,
 *        based on its operation type (`node->op`) and its input tensors.
 *
 *  Purpose:
 *      - This function allows recomputation of node outputs when they
 *        have been deleted or released during checkpointing.
 *      - It’s also used for lazy forward evaluation, debug visualization,
 *        or runtime validation of the computational graph.
 *
 *  Core logic:
 *      1️⃣  Validate that the node exists.
 *      2️⃣  Switch over the node’s operation (`Op` enum).
 *      3️⃣  Retrieve the node’s input tensors (`node->inputs[i]->value`).
 *      4️⃣  Perform the appropriate mathematical operation.
 *      5️⃣  Return the computed output tensor.
 *      6️⃣  If unsupported, throw a runtime error.
 */
// Tensor forward_eval_node(const std::shared_ptr<Node> &node) {
//     if (!node)
//         throw std::runtime_error("forward_eval_node: null node");

//     switch (node->op) {


//         case Op::Add: {
//             const Tensor &A = node->inputs[0]->value;
//             const Tensor &B = node->inputs[1]->value;

//             if (A.rows() == 0 || B.rows() == 0 || A.cols() == 0 || B.cols() == 0) {
//                 std::cerr << "\n[DEBUG] Shape mismatch detected in Add op!\n";
//                 std::cerr << "Node@" << node.get() << " name=\"" 
//                         << (node->debug_name ? node->debug_name : "(null)") << "\"\n";
//                 std::cerr << "  Input0 shape: " << A.rows() << "x" << A.cols()
//                         << " ptr=" << A.data() << "\n";
//                 std::cerr << "  Input1 shape: " << B.rows() << "x" << B.cols()
//                         << " ptr=" << B.data() << "\n";
//                 std::cerr << "  Input0 node@" << node->inputs[0].get()
//                         << " checkpoint=" << node->inputs[0]->is_checkpoint << "\n";
//                 std::cerr << "  Input1 node@" << node->inputs[1].get()
//                         << " checkpoint=" << node->inputs[1]->is_checkpoint << "\n";
//                 throw std::runtime_error("add_: shape mismatch");
//             }

//             return A + B;
//         }

//         case Op::Sub: {
//             const Tensor &A = node->inputs[0]->value;
//             const Tensor &B = node->inputs[1]->value;
//             return A - B; // elementwise subtraction
//         }
//         case Op::Mul: {
//             const Tensor &A = node->inputs[0]->value;
//             const Tensor &B = node->inputs[1]->value;
//             return A * B; // elementwise multiplication
//         }

//         case Op::MatMul: {
//             const Tensor &A = node->inputs[0]->value;
//             const Tensor &B = node->inputs[1]->value;
//             return Tensor::matmul(A, B);
//         }

//         // ============================================================
//         // Unary elementwise activations
//         // ============================================================
//         case Op::Relu: {
//             const Tensor &X = node->inputs[0]->value;
//             return Tensor::relu(X);
//         }
//         case Op::Sigmoid: {
//             const Tensor &X = node->inputs[0]->value;
//             return Tensor::sigmoid(X);
//         }
//         case Op::Tanh: {
//             const Tensor &X = node->inputs[0]->value;
//             return Tensor::tanh(X);
//         }
//         case Op::Exp: {
//             const Tensor &X = node->inputs[0]->value;
//             return Tensor::exp(X);
//         }
//         case Op::Log: {
//             const Tensor &X = node->inputs[0]->value;
//             return Tensor::log(X);
//         }
//         case Op::Sum: {
//             const Tensor &X = node->inputs[0]->value;
//             return Tensor::sum_all(X);
//         }

//         case Op::AlibiAttention: {
//             const Tensor &a = node->inputs[0]->value;
//             const Tensor &b = node->inputs[1]->value;
//             const Tensor &c = node->inputs[2]->value;
//             const Tensor &d = node->inputs[3]->value;

//             // Step 1: compute projections
//             Tensor q = Tensor::matmul(a, b);
//             Tensor k = Tensor::matmul(a, c);
//             Tensor v = Tensor::matmul(a, d);

//             // Step 2: scaled dot-product attention
//             Tensor logits = Tensor::matmul(q, Tensor::transpose(k) * (1.f / sqrt(float(k.cols()))));

//             // Step 3: add ALIBI bias (creates a position-dependent attention slope)
//             Tensor bias = Tensor::alibi(logits.rows(), logits.cols(), /*m*/128);
//             Tensor g = logits + bias;

//             // Step 4: softmax normalization over rows
//             Tensor s = Tensor::softmax_row(g);

//             // Step 5: output = attention weights × values
//             Tensor y = Tensor::matmul(s, v);
//             return y;
//         }

//                 case Op::Leaf:
//             return node->value;

//                default:
//             if (!node->tape.empty()) {
//                 return *(node->tape.back());
//             }
//             throw std::runtime_error("forward_eval_node: unsupported op for recompute");
//     }
// }


// Tensor forward_eval_node(Node* node) {
//     // Non-owning shared_ptr wrapper — prevents deletion of node
//     return forward_eval_node(std::shared_ptr<Node>(node, [](Node*){}));
// }

// Robust forward evaluator for a single node (recompute-friendly)
Tensor forward_eval_node(const std::shared_ptr<Node> &node) {
    if (!node) throw std::runtime_error("forward_eval_node: null node");

    auto dbg_name = node->debug_name ? node->debug_name : "(null)";
    try {
        switch (node->op) {

        case Op::Leaf:
            // Leaf nodes simply hold their value
            return node->value;

        case Op::Add: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            if (A.size() == 0 || B.size() == 0) {
                std::ostringstream ss;
                ss << "[forward_eval_node::Add] empty input for node@" << node.get()
                   << " name=\"" << dbg_name << "\" A.size=" << A.size() << " B.size=" << B.size();
                throw std::runtime_error(ss.str());
            }
            return A + B;
        }

        case Op::Sub: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return A - B;
        }

        case Op::Mul: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return A * B;
        }

        case Op::MatMul: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            if (A.cols() != B.rows()) {
                std::ostringstream ss;
                ss << "matmul: inner dim mismatch for node@" << node.get()
                   << " name=\"" << dbg_name << "\" A=" << A.rows() << "x" << A.cols()
                   << " B=" << B.rows() << "x" << B.cols();
                throw std::runtime_error(ss.str());
            }
            return Tensor::matmul(A, B);
        }

        // Elementwise unary ops
        case Op::Relu: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::relu(X);
        }
        case Op::Sigmoid: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::sigmoid(X);
        }
        case Op::Tanh: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::tanh(X);
        }
        case Op::Exp: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::exp(X);
        }
        case Op::Log: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::log(X);
        }

        case Op::Sum: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::sum_all(X);
        }

        case Op::Transpose: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::transpose(X);
        }

        case Op::SoftmaxRow: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::softmax_row(X);
        }
        case Op::Softplus: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::softplus(X);
        }
        // case Op::LayerNorm: {
        //     // Assume layernorm(input) – returns same shape
        //     const Tensor &X = node->inputs[0]->value;
        //     return Tensor::rmsn(X);
        // }

        case Op::RMSNorm: {
            const Tensor &X = node->inputs[0]->value;
                return Tensor::rmsn(X);
        }

        case Op::Attention:
        case Op::AlibiAttention: {
            // Generic attention implementation that works on 2D flattened (B*S, E)
            // Expected inputs: (Q_src, K_src, V_src, Wout)
            // If your Attention op expects other shapes, adjust accordingly.
            const Tensor &Q = node->inputs[0]->value;
            const Tensor &K = node->inputs[1]->value;
            const Tensor &V = node->inputs[2]->value;
            const Tensor &Wo = node->inputs[3]->value; // output projection, optional

            // Basic checks
            if (Q.size() == 0 || K.size() == 0 || V.size() == 0)
                throw std::runtime_error("forward_eval_node::Attention: Q/K/V empty");

            // scores = Q @ K^T
            Tensor Kt = Tensor::transpose(K);
            if (Q.cols() != Kt.rows()) {
                std::ostringstream ss;
                ss << "attention: inner dim mismatch Q("<<Q.rows()<<"x"<<Q.cols()
                   <<") K^T("<<Kt.rows()<<"x"<<Kt.cols()<<") for node@"<<node.get();
                throw std::runtime_error(ss.str());
            }
            Tensor logits = Tensor::matmul(Q, Kt);

            // scale by sqrt(d_k) if desired (simple heuristic)
            float scale = 1.0f / std::sqrt(float(std::max(1, K.cols())));
            if (scale != 1.0f) logits = logits * scale;

            // If ALIBI variant, add bias (use Tensor::alibi if available)
            if (node->op == Op::AlibiAttention) {
                Tensor bias = Tensor::alibi(logits.rows(), logits.cols(), /*m=*/128);
                logits = logits + bias;
            }

            Tensor weights = Tensor::softmax_row(logits);
            Tensor context = Tensor::matmul(weights, V);

            if (Wo.size() != 0) {
                if (context.cols() != Wo.rows()) {
                    std::ostringstream ss;
                    ss << "attention: Wo shape mismatch context.cols="<<context.cols()
                       << " Wo.rows="<<Wo.rows()<<" for node@"<<node.get();
                    throw std::runtime_error(ss.str());
                }
                return Tensor::matmul(context, Wo);
            }
            return context;
        }

        case Op::GELU: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::gelu_tanh(X);
        }
        case Op::LeakyRelu: {
            const Tensor &X = node->inputs[0]->value;
            float alpha = 0.01f; // default slope
            if (node->inputs.size() > 1) {
                const Tensor &AlphaTensor = node->inputs[1]->value;
                if (AlphaTensor.numel() != 1) {
                    std::ostringstream ss;
                    ss << "leaky_relu: alpha must be scalar for node@" << node.get()
                       << " name=\"" << dbg_name << "\"";
                    throw std::runtime_error(ss.str());
                }
                alpha = AlphaTensor.data()[0];
            }
            return Tensor::leaky_relu(X, alpha);
        }
        
        // Fallback: if node has a tape with a saved forward result, return it
        default:
            if (!node->tape.empty() && node->tape.back()) {
                return *(node->tape.back());
            }
            // Unknown op: fail explicitly (useful message)
            std::ostringstream ss;
            ss << "forward_eval_node: unsupported op for recompute (op=" << static_cast<int>(node->op)
               << ") node@" << node.get() << " name=\"" << dbg_name << "\"";
            throw std::runtime_error(ss.str());
        } // switch
    } catch (const std::exception &e) {
        std::ostringstream ss;
        ss << "[forward_eval_node] exception for node@" << node.get()
           << " name=\"" << dbg_name << "\": " << e.what();
        std::cerr << ss.str() << std::endl;
        throw; // rethrow to let caller handle
    }
}

// convenience wrapper accepting Node*
Tensor forward_eval_node(Node* node) {
    // wrap non-owning pointer to avoid changing ownership / lifetime
    return forward_eval_node(std::shared_ptr<Node>(node, [](Node*){}));
}

// -----------------------------------------------------------------------------
// checkpoint() — Mark a node for checkpointing
// -----------------------------------------------------------------------------

/*
 * checkpoint():
 * --------------
 *  A user-facing function that marks a value (and its corresponding node)
 *  for checkpointing.
 *
 *  When a node is checkpointed:
 *      - Its intermediate activations may be freed to save memory.
 *      - During backpropagation, if its output is required,
 *        the system will recompute it using `forward_eval_node()`
 *        and its input dependencies.
 *
 *  Parameters:
 *      - v    : Value object wrapping the Node to be checkpointed.
 *      - opts : CheckpointOptions structure (default-initialized).
 *
 *  Returns:
 *      - The same Value `v` (allowing function chaining).
 *
 *  Internally, it calls:
 *      `checkpoint_impl::mark_node_checkpoint()`
 *  which performs the actual checkpoint marking and state saving.
 *
 *  Example usage:
 *      Value y = checkpoint(forward_pass(x));
 *      Tensor loss = mse(y, target);
 *      backward(loss);
 */
Value checkpoint(const Value &v, const CheckpointOptions &opts) {
    if (!v.node) return v; // safety check: skip if Value has no node
    ag::checkpoint_impl::mark_node_checkpoint(v.node, opts);
    return v;
}

} // namespace ag