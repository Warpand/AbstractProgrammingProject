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
    AutoGrad(F& data, bool requires_grad = false)
        : node(std::make_shared<Node<F>>(data, requires_grad)) {}

    AutoGrad(F&& data, bool requires_grad = false)
        : node(std::make_shared<Node<F>>(data, requires_grad)) {}

    explicit AutoGrad(std::shared_ptr<Node<F>>& node) : node(node) {}

    explicit AutoGrad(std::shared_ptr<Node<F>>&& node) : node(std::move(node)) {}

    void connect(const AutoGrad& other) { node->add_edge(other.node); }

    [[nodiscard]] F& data() { return node->data(); }

    [[nodiscard]] const F& data() const { return node->data(); }

    [[nodiscard]] const F& grad() const { return node->get_grad(); }

    [[nodiscard]] bool requires_grad() const { return node->requires_backward(); }

    [[nodiscard]] bool has_grad() const { return node->has_grad(); }

    void backward() { node->backward(); }

    AutoGrad copy(bool requires_grad = false) {
        return AutoGrad(node->data(), requires_grad);
    }

    void set_requires_grad(bool value) { node->set_requires_grad(value); }
};

template <Field F, template <Field> typename AutoGradFunc>
class Function {
   public:
    static AutoGrad<F> call(const AutoGrad<F>& arg) {
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

template <Field F, template <Field> typename AutoGradBiFunc>
class BiFunction {
   public:
    static AutoGrad<F> call(const AutoGrad<F>& x, const AutoGrad<F>& y) {
        F func_output = AutoGradBiFunc<F>::forward(x.data(), y.data());
        auto node = std::make_shared<Node<F>>(std::move(func_output));
        AutoGrad<F> result(node);
        if (GradContext<F>::grad_enabled()
            && (x.requires_grad() || y.requires_grad())) {
            result.connect(x);
            result.connect(y);
            node->set_backward_func(
                std::make_unique<BinaryBackwardFunc<F>>(AutoGradBiFunc<F>::backward)
            );
        }
        return result;
    }
};

template <Field F, int NUM_ARGS, template <Field, int> typename AutoGradMultiFunc>
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
        F func_output = AutoGradMultiFunc<F, NUM_ARGS>::forward(func_args);
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
                    AutoGradMultiFunc<F, NUM_ARGS>::backward
                )
            );
        }
        return result;
    }
};

template <Field F, typename ScalarType, template <Field> typename AutoGradScalarFunc>
class ScalarFunction {
   public:
    static AutoGrad<F> call(const AutoGrad<F>& arg, ScalarType scalar) {
        F func_output = AutoGradScalarFunc<F>::forward(arg.data(), scalar);
        auto node = std::make_shared<Node<F>>(std::move(func_output));
        AutoGrad<F> result(node);
        if (GradContext<F>::grad_enabled() && arg.requires_grad()) {
            result.connect(arg);
            node->set_backward_func(std::make_unique<ScalarBackwardFunc<F, ScalarType>>(
                AutoGradScalarFunc<F>::backward, scalar
            ));
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

template <Field F>
class Mul : public BiFunction<F, Mul> {
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
class Add : public BiFunction<F, Add> {
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
class Subtract : public BiFunction<F, Subtract> {
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
class Div : public BiFunction<F, Div> {
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
class Pow : public ScalarFunction<F, int, Pow> {
    static F _call(typename FieldTraits<F>::arg_type x, const int exp) {
        if (exp >= 0)
            return Pow::_pow(x, exp);
        return FieldTraits<F>::reverse(Pow::_pow(x, exp));
    }

    static F _pow(F x, int exp) {
        F r = FieldTraits<F>::one;
        while (exp > 0) {
            if (exp & 1) {
                if constexpr (HasInPlaceOperators<F>)
                    r *= x;
                else
                    r = r * x;
            }
            if constexpr (HasInPlaceOperators<F>)
                x *= x;
            else
                x = x * x;
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
class FlipSign : public Function<F, FlipSign> {
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
