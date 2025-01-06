#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "concepts.h"
#include "context.h"
#include "graph.h"

namespace autograd {
template <Field F>
class AutoGrad {
    std::shared_ptr<Node<F>> node;

   public:
    explicit AutoGrad(F& data, bool requires_grad = false)
        : node(std::make_shared<Node<F>>(data, requires_grad)) {}

    explicit AutoGrad(F&& data, bool requires_grad = false)
        : node(std::make_shared<Node<F>>(data, requires_grad)) {}

    explicit AutoGrad(std::shared_ptr<Node<F>>& node) : node(node) {}

    explicit AutoGrad(std::shared_ptr<Node<F>>&& node) : node(std::move(node)) {}

    void connect(AutoGrad& other) { node->add_edge(other.node); }

    F& data() { return node->data(); }

    const F& data() const { return node->data(); }

    const F& grad() const { return node->get_grad(); }

    [[nodiscard]] bool requires_grad() const { return node->requires_backward(); }

    void backward() { node->backward(); }

    AutoGrad copy(bool requires_grad = false) {
        return AutoGrad(node->data(), requires_grad);
    }

    void set_requires_grad(bool value) {}
};

template <Field F, template <Field> typename AutoGradFunc>
class Function {
   public:
    static AutoGrad<F> call(AutoGrad<F> arg) {
        F func_output = AutoGradFunc<F>::forward(arg.data());
        auto node = std::make_shared<Node<F>>(std::move(func_output));
        AutoGrad<F> result(node);
        if (GradContext<F>::grad_enabled() && arg.requires_grad()) {
            result.connect(arg);
            node->set_backward_func(
                std::make_unique<UnaryBackwardFunc<F>>(AutoGradFunc<F>::backward)
            );
        }
        return result;
    }
};

template <Field F>
class Identity : public Function<F, Identity> {
   public:
    static F forward(typename FieldTraits<F>::arg_type x) { return x; }

    static F backward(typename FieldTraits<F>::arg_type) { return FieldTraits<F>::one; }
};

template <>
class FieldTraits<double> {
   public:
    typedef double arg_type;
    constexpr static double one = 1.0;
};
}  // namespace autograd

#endif  // AUTOGRAD_H
