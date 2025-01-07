#ifndef CONCEPTS_H
#define CONCEPTS_H

#include <concepts>

namespace autograd {
template <typename T>
class FieldTraits {
   public:
    typedef T& arg_type;
};

template <typename T>
concept Field = requires(T x, T y) {
    { x + y };
    // clang-format off
    { x * y };
    // clang-format on
    { x - y };
    { x / y };
    { x += y };
    { -x };
    { FieldTraits<T>::one };
    { FieldTraits<T>::reverse(x) };
    typename FieldTraits<T>::arg_type;
};

template <typename T>
concept HasInPlaceOperators = requires(T x, T y) {
    { x += y };
    { x *= y };
    { x -= y };
    { x /= y };
};
}  // namespace autograd

#endif  // CONCEPTS_H
