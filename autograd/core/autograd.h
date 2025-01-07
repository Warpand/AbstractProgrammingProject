#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <algorithm>
#include <ostream>

#include "concepts.h"
#include "context.h"
#include "graph.h"

namespace autograd {
template <Field F>
class AutoGrad {
    std::shared_ptr<Node<F>> node;

   public:
    AutoGrad(const F& data, bool requires_grad = false)
        : node(std::make_shared<Node<F>>(data, requires_grad)) {}

    AutoGrad(F&& data, bool requires_grad = false)
        : node(std::make_shared<Node<F>>(data, requires_grad)) {}

    explicit AutoGrad(const std::shared_ptr<Node<F>>& node) : node(node) {}

    explicit AutoGrad(std::shared_ptr<Node<F>>&& node) : node(std::move(node)) {}

    void connect(const AutoGrad& other) const { node->add_edge(other.node); }

    [[nodiscard]] F& data() { return node->data(); }

    [[nodiscard]] const F& data() const { return node->data(); }

    [[nodiscard]] const F& grad() const { return node->get_grad(); }

    [[nodiscard]] bool requires_grad() const { return node->requires_backward(); }

    [[nodiscard]] bool has_grad() const { return node->has_grad(); }

    void backward() const { node->backward(); }

    AutoGrad copy(bool requires_grad = false) const {
        return AutoGrad(node->data(), requires_grad);
    }

    void set_requires_grad(bool value) { node->set_requires_grad(value); }
};

template <Field F, typename AutoGradFunc>
class Function {
   public:
    static AutoGrad<F> call(const AutoGrad<F>& arg) {
        F func_output = AutoGradFunc::forward(arg.data());
        auto node = std::make_shared<Node<F>>(std::move(func_output));
        AutoGrad<F> result(node);
        if (GradContext<F>::grad_enabled() && arg.requires_grad()) {
            result.connect(arg);
            node->set_backward_func(
                std::make_unique<UnaryBackwardFunc<F>>(AutoGradFunc::backward)
            );
        }
        return result;
    }
};

template <Field F, typename AutoGradBiFunc>
class BiFunction {
   public:
    static AutoGrad<F> call(const AutoGrad<F>& x, const AutoGrad<F>& y) {
        F func_output = AutoGradBiFunc::forward(x.data(), y.data());
        auto node = std::make_shared<Node<F>>(std::move(func_output));
        AutoGrad<F> result(node);
        if (GradContext<F>::grad_enabled()
            && (x.requires_grad() || y.requires_grad())) {
            result.connect(x);
            result.connect(y);
            node->set_backward_func(
                std::make_unique<BinaryBackwardFunc<F>>(AutoGradBiFunc::backward)
            );
        }
        return result;
    }
};

template <Field F, int NUM_ARGS, typename AutoGradMultiFunc>
class MultiFunction {
    static_assert(
        NUM_ARGS > 0,
        "autograd::MultiFunction must take more than 0 arguments."
    );
    static_assert(
        NUM_ARGS != 1,
        "autograd::MultiFunction used for a single argument function - use "
        "autograd::Function instead."
    );
    static_assert(
        NUM_ARGS != 2,
        "autograd::MultiFunction used for a binary function - use autograd::BiFunction "
        "instead."
    );

   public:
    static AutoGrad<F> call(std::array<const AutoGrad<F>&, NUM_ARGS> args) {
        std::array<typename FieldTraits<F>::arg_type, NUM_ARGS> func_args;
        for (int i = 0; i < NUM_ARGS; i++)
            func_args[i] = args[i].data();
        F func_output = AutoGradMultiFunc::forward(func_args);
        auto node = std::make_shared<Node<F>>(std::move(func_output));
        AutoGrad<F> result(node);
        if (GradContext<F>::grad_enabled()
            && std::any_of(args.begin(), args.end(), [](const AutoGrad<F>& arg) {
                   return arg.requires_grad();
               })) {
            for (AutoGrad<F>& other : args)
                result.connect(other);
            node->set_backward_func(
                std::make_unique<MultiArgBackwardFunction<F, NUM_ARGS>>(
                    AutoGradMultiFunc::backward
                )
            );
        }
        return result;
    }
};

template <Field F, typename ScalarType, typename AutoGradScalarFunc>
class ScalarFunction {
   public:
    static AutoGrad<F> call(const AutoGrad<F>& arg, ScalarType scalar) {
        F func_output = AutoGradScalarFunc::forward(arg.data(), scalar);
        auto node = std::make_shared<Node<F>>(std::move(func_output));
        AutoGrad<F> result(node);
        if (GradContext<F>::grad_enabled() && arg.requires_grad()) {
            result.connect(arg);
            node->set_backward_func(std::make_unique<ScalarBackwardFunc<F, ScalarType>>(
                AutoGradScalarFunc::backward, scalar
            ));
        }
        return result;
    }
};

template <Field F>
class Identity : public Function<F, Identity<F>> {
   public:
    static F forward(typename FieldTraits<F>::arg_type x) { return x; }

    static F backward(typename FieldTraits<F>::arg_type) { return FieldTraits<F>::one; }
};

template <Field F>
class Mul : public BiFunction<F, Mul<F>> {
   public:
    static F
    forward(typename FieldTraits<F>::arg_type x, typename FieldTraits<F>::arg_type y) {
        return x * y;
    }

    static std::pair<F, F>
    backward(typename FieldTraits<F>::arg_type x, typename FieldTraits<F>::arg_type y) {
        return {y, x};
    }
};

template <Field F>
class Add : public BiFunction<F, Add<F>> {
   public:
    static F
    forward(typename FieldTraits<F>::arg_type x, typename FieldTraits<F>::arg_type y) {
        return x + y;
    }

    static std::pair<F, F>
    backward(typename FieldTraits<F>::arg_type x, typename FieldTraits<F>::arg_type y) {
        return {FieldTraits<F>::one, FieldTraits<F>::one};
    }
};

template <Field F>
class Subtract : public BiFunction<F, Subtract<F>> {
   public:
    static F
    forward(typename FieldTraits<F>::arg_type x, typename FieldTraits<F>::arg_type y) {
        return x - y;
    }

    static std::pair<F, F>
    backward(typename FieldTraits<F>::arg_type x, typename FieldTraits<F>::arg_type y) {
        return {FieldTraits<F>::one, -FieldTraits<F>::one};
    }
};

template <typename F>
class Div : public BiFunction<F, Div<F>> {
   public:
    static F
    forward(typename FieldTraits<F>::arg_type x, typename FieldTraits<F>::arg_type y) {
        return x / y;
    }

    static std::pair<F, F>
    backward(typename FieldTraits<F>::arg_type x, typename FieldTraits<F>::arg_type y) {
        return {FieldTraits<F>::reverse(y), -(x / (y * y))};
    }
};

template <Field F>
class Pow : public ScalarFunction<F, int, Pow<F>> {
    static F _call(typename FieldTraits<F>::arg_type x, const int exp) {
        if (exp >= 0)
            return Pow::_pow(x, exp);
        return FieldTraits<F>::reverse(Pow::_pow(x, exp));
    }

    static F _pow(F x, int exp) {
        F r = FieldTraits<F>::one;
        while (exp > 0) {
            if (exp & 1)
                r *= x;
            x *= x;
            exp >>= 1;
        }
        return r;
    }

   public:
    static F forward(typename FieldTraits<F>::arg_type x, int exp) {
        return Pow::_call(x, exp);
    }

    static F backward(typename FieldTraits<F>::arg_type x, int exp) {
        return exp * Pow::_call(x, exp - 1);
    }
};

template <typename F>
class FlipSign : public Function<F, FlipSign<F>> {
   public:
    static F forward(typename FieldTraits<F>::arg_type x) { return -x; }

    static F backward(typename FieldTraits<F>::arg_type) {
        return -FieldTraits<F>::one;
    }
};

template <Field F>
AutoGrad<F> operator+(const AutoGrad<F>& x, const AutoGrad<F>& y) {
    return Add<F>::call(x, y);
}

template <Field F>
AutoGrad<F> operator-(const AutoGrad<F>& x, const AutoGrad<F>& y) {
    return Subtract<F>::call(x, y);
}

template <Field F>
AutoGrad<F> operator*(const AutoGrad<F>& x, const AutoGrad<F>& y) {
    return Mul<F>::call(x, y);
}

template <Field F>
AutoGrad<F> operator/(const AutoGrad<F>& x, const AutoGrad<F>& y) {
    return Div<F>::call(x, y);
}

template <Field F>
AutoGrad<F> operator-(const AutoGrad<F>& x) {
    return FlipSign<F>::call(x);
}

template <Field F>
std::ostream& operator<<(std::ostream& stream, const AutoGrad<F>& x) {
    stream << "data: " << x.data() << " grad: ";
    if (x.has_grad())
        stream << x.grad();
    else
        stream << "None";
    return stream;
}

template <>
class FieldTraits<double> {
   public:
    typedef double arg_type;
    constexpr static double one = 1.0;
    static double reverse(const double x) { return 1.0 / x; }
};
}  // namespace autograd

#endif  // AUTOGRAD_H
