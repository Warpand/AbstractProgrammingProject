#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <functional>

#include "concepts.h"
#include "context.h"
#include "graph.h"

namespace autograd {
template <Field F>
class AutoGrad {
    std::shared_ptr<Node<F>> node;

   public:
    AutoGrad(F& data, bool requires_grad = false)
        : node(std::make_shared<Node<F>>(data, requires_grad)) {}

    AutoGrad(F&& data, bool requires_grad = false)
        : node(std::make_shared<Node<F>>(data, requires_grad)) {}

    std::shared_ptr<Node<F>> get_node() { return node; }

    F& data() { return node->data(); }

    const F& data() const { return node->data(); }

    const F& grad() const { return node->get_grad(); }

    [[nodiscard]] bool requires_grad() const { return node->get_requires_grad(); }

    [[nodiscard]] bool is_leaf() const { return node->is_leaf(); }

    void backward() { node->backward(); }
};

template <Field F, template <Field> typename AutoGradFunc>
class Function {
    class SingleEdge final : public Edge<F> {
        typedef std::function<
            typename FieldTraits<F>::arg_type(typename FieldTraits<F>::arg_type)>
            BackwardFuncType;
        BackwardFuncType backward_func;

       public:
        SingleEdge(std::shared_ptr<Node<F>> target, BackwardFuncType backward_func)
            : Edge<F>(target), backward_func(backward_func) {}

        void backward(typename FieldTraits<F>::arg_type source_grad) override {
            Edge<F>::pass_to_target(
                backward_func(Edge<F>::get_target()->data()), source_grad
            );
        }
    };

   public:
    static AutoGrad<F> call(AutoGrad<F>& arg) {
        F func_output = AutoGradFunc<F>::forward(arg.data());
        AutoGrad<F> result(std::move(func_output));
        if (GradContext<F>::grad_enabled() && (!arg.is_leaf() || arg.requires_grad())) {
            result.get_node()->add_edge(
                std::make_unique<SingleEdge>(arg.get_node(), AutoGradFunc<F>::backward)
            );
        }
        return result;
    }
};

template <Field F>
class Identity : public Function<F, Identity> {
   public:
    static F forward(typename FieldTraits<F>::arg_type x) { return x; }

    static F backward(typename FieldTraits<F>::arg_type x) {
        return FieldTraits<F>::one;
    }
};

template <>
class FieldTraits<double> {
   public:
    typedef double arg_type;
    constexpr static double one = 1.0;
};
}  // namespace autograd

#endif  // AUTOGRAD_H
