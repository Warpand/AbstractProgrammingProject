#ifndef GRAPH_H
#define GRAPH_H

#include <functional>
#include <memory>
#include <ranges>
#include <unordered_set>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "concepts.h"
#include "constants.h"

namespace autograd {
template <Field F>
class Node;

template <Field F>
class BackwardFunc {
   protected:
    static void pass_to_target(
        Node<F>* target,
        typename FieldTraits<F>::arg_type target_grad,
        typename FieldTraits<F>::arg_type source_grad
    ) {
        if (target->requires_backward())
            target->accumulate_grad(target_grad * source_grad);
    }

   public:
    virtual void backward(
        typename Node<F>::BackwardEdges& targets,
        typename FieldTraits<F>::arg_type source_grad
    ) = 0;
    virtual ~BackwardFunc() = default;
};

template <Field F>
class Node {
   public:
    typedef boost::container::small_vector<std::shared_ptr<Node>, INLINE_EDGE_CAPACITY>
        BackwardEdges;

   private:
    constexpr static auto SECOND_PASS_ERR_MSG =
        "Trying to backward through the graph a second time.";
    constexpr static auto BACKWARD_ERR_MSG =
        "Calling backward on a node that does not require grad and has no backward "
        "function defined.";
    constexpr static auto SET_REQUIRES_GRAD_ERR_MSG =
        "Changing requires_grad is possible only for leaf nodes.";

    F _data;
    bool requires_grad;
    std::unique_ptr<F> grad = nullptr;
    std::unique_ptr<BackwardFunc<F>> backward_func = nullptr;
    BackwardEdges backward_edges;

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

    void pre_backward() {
        if (!is_leaf() && backward_edges.empty())
            throw std::runtime_error(SECOND_PASS_ERR_MSG);
    }

    void do_backward() { backward_func->backward(backward_edges, *grad); }

    void post_backward() {
        if (!requires_grad)
            grad = nullptr;
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
        if (!requires_backward())
            throw std::runtime_error(BACKWARD_ERR_MSG);
        std::vector<Node*> order = topological_sort();
        grad = std::make_unique<F>(FieldTraits<F>::one);
        for (Node* node :
             order | std::views::filter([](const Node* n) { return !n->is_leaf(); })) {
            node->pre_backward();
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

    void set_backward_func(std::unique_ptr<BackwardFunc<F>>&& func) {
        backward_func = std::move(func);
    }

    void set_requires_grad(const bool value) {
        if (!is_leaf())
            throw std::runtime_error(SET_REQUIRES_GRAD_ERR_MSG);
        requires_grad = value;
    }

    [[nodiscard]] bool is_leaf() const { return backward_func == nullptr; }

    [[nodiscard]] bool requires_backward() const { return !is_leaf() || requires_grad; }

    [[nodiscard]] F& data() { return _data; }

    [[nodiscard]] const F& data() const { return _data; }

    [[nodiscard]] const F& get_grad() const { return *grad; }
};

template <Field F>
class UnaryBackwardFunc final : public BackwardFunc<F> {
    // clang-format off
    typedef std::function<
        typename FieldTraits<F>::arg_type(typename FieldTraits<F>::arg_type)
    > FuncType;
    // clang-format on
    FuncType func;

   public:
    explicit UnaryBackwardFunc(FuncType func) : func(func) {}

    void backward(
        typename Node<F>::BackwardEdges& targets,
        typename FieldTraits<F>::arg_type source_grad
    ) override {
        BackwardFunc<F>::pass_to_target(
            targets[0].get(), func(targets[0]->data()), source_grad
        );
    }
};
}  // namespace autograd

#endif  // GRAPH_H
