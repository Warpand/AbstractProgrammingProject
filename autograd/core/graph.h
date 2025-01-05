#ifndef GRAPH_H
#define GRAPH_H

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "concepts.h"
#include "constants.h"

namespace autograd {
template <Field F>
class Node {
    constexpr static auto BACKWARD_ERR_MSG =
        "Error: calling backward on a leaf node.\n Backward was called on a user "
        "created leaf node or a node that has already passed its gradient.";

    F _data;
    bool requires_grad;
    std::unique_ptr<F> grad = nullptr;

   protected:
    boost::container::small_vector<std::shared_ptr<Node>, INLINE_EDGE_CAPACITY>
        backward_edges;

   private:
    void topological_sort_recursion(
        std::vector<Node*>& result,
        std::unordered_set<Node*>& visited
    ) {
        visited.insert(this);
        for (size_t i = 0; i < backward_edges.size(); i++) {
            if (!visited.contains(backward_edges[i].get()))
                backward_edges[i]->topological_sort_recursion(result, visited);
        }
        result.push_back(this);
    }

    std::vector<Node*> topological_sort() {
        std::vector<Node*> result;
        std::unordered_set<Node*> visited;
        topological_sort_recursion(result, visited);
        std::reverse(result.begin(), result.end());
        return result;
    }

    void post_backward() {
        if (!requires_grad)
            grad = nullptr;
    }

   protected:
    virtual void do_backward() = 0;

    void pass_backward(Node* target, typename FieldTraits<F>::arg_type target_grad) {
        target->accumulate_grad(*grad * target_grad);
    }

   public:
    explicit Node(F& data, const bool requires_grad = false)
        : _data(data), requires_grad(requires_grad) {}

    explicit Node(F&& data, const bool requires_grad = false)
        : _data(std::move(data)), requires_grad(requires_grad) {}

    void add_edge(std::shared_ptr<Node>& edge) { backward_edges.push_back(edge); }

    void add_edge(std::shared_ptr<Node>&& edge) {
        backward_edges.push_back(std::move(edge));
    }

    void backward() {
        if (is_leaf())
            throw std::runtime_error(BACKWARD_ERR_MSG);
        std::vector<Node*> order = topological_sort();
        grad = std::make_unique<F>(FieldTraits<F>::one);
        for (Node* node : order) {
            node->do_backward();
            node->post_backward();
        }
        for (Node* node : order)
            node->backward_edges.clear();  // "garbage collect"
    }

    void accumulate_grad(typename FieldTraits<F>::arg_type passed_value) {
        if (grad == nullptr)
            grad = std::make_unique<F>(passed_value);
        else
            *grad += passed_value;
    }

    [[nodiscard]] bool is_leaf() const { return backward_edges.empty(); }

    [[nodiscard]] bool get_requires_grad() const { return requires_grad; }

    [[nodiscard]] F& data() { return _data; }

    [[nodiscard]] const F& data() const { return _data; }

    [[nodiscard]] const F& get_grad() const { return *grad; }

    virtual ~Node() = default;
};

template <Field F>
class LeafNode final : public Node<F> {
   public:
    using Node<F>::Node;

    void do_backward() override {}
};

template <Field F>
class UnaryNode final : public Node<F> {
    // clang-format off
    typedef std::function<
        typename FieldTraits<F>::arg_type(typename FieldTraits<F>::arg_type)
    > BackwardFuncType;
    // clang-format on
    BackwardFuncType backward_func;

   public:
    UnaryNode(F& data, BackwardFuncType backward_func)
        : Node<F>(data), backward_func(backward_func) {}

    UnaryNode(F&& data, BackwardFuncType backward_func)
        : Node<F>(std::move(data)), backward_func(backward_func) {}

    void do_backward() override {
        Node<F>::pass_backward(
            Node<F>::backward_edges[0].get(), backward_func(Node<F>::data())
        );
    }
};
}  // namespace autograd

#endif  // GRAPH_H
