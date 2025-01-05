#ifndef GRAPH_H
#define GRAPH_H

#include <memory>
#include <unordered_set>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "concepts.h"
#include "constants.h"

namespace autograd {
template <Field F>
class Node;

template <Field F>
class Edge {
    std::shared_ptr<Node<F>> target;

   protected:
    void pass_to_target(
        typename FieldTraits<F>::arg_type target_grad,
        typename FieldTraits<F>::arg_type source_grad
    ) {
        target->accumulate_grad(target_grad * source_grad);
    }

   public:
    explicit Edge(std::shared_ptr<Node<F>> target) : target(target) {}

    Node<F>* get_target() { return target.get(); }

    virtual void backward(typename FieldTraits<F>::arg_type source_grad) = 0;

    virtual ~Edge() = default;
};

template <Field F>
class Node {
    F _data;
    bool requires_grad;
    std::unique_ptr<F> grad = nullptr;
    boost::container::small_vector<std::unique_ptr<Edge<F>>, INLINE_EDGE_CAPACITY>
        backward_edges;

    void topological_sort_recursion(
        std::vector<Node*>& result,
        std::unordered_set<Node*>& visited
    ) {
        visited.insert(this);
        for (size_t i = 0; i < backward_edges.size(); i++) {
            Node* target = backward_edges[i]->get_target();
            if (!visited.contains(target))
                target->topological_sort_recursion(result, visited);
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

    void do_backward() {
        for (size_t i = 0; i < backward_edges.size(); i++)
            backward_edges[i]->backward(*grad);
        if (!requires_grad)
            grad = nullptr;
    }

   public:
    explicit Node(F& data, const bool requires_grad = false)
        : _data(data), requires_grad(requires_grad) {}

    explicit Node(F&& data, const bool requires_grad = false)
        : _data(data), requires_grad(requires_grad) {}

    void add_edge(std::unique_ptr<Edge<F>>&& edge) {
        backward_edges.push_back(std::move(edge));
    }

    void backward() {
        std::vector<Node*> order = topological_sort();
        grad = std::make_unique<F>(FieldTraits<F>::one);
        for (Node* node : order)
            node->do_backward();
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
};
}  // namespace autograd

#endif  // GRAPH_H
