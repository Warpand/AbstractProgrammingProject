#ifndef CONCEPTS_H
#define CONCEPTS_H

#include <concepts>
#include <type_traits>

namespace autograd {
template <typename T>
class FieldTraits {
    typedef T& arg_type;
};

template <typename T>
concept Field = requires(T x, T y) {
    { x + y };
    { x* y };
    { x - y };
    { x / y };
    { x += y };
    { FieldTraits<T>::one };
    typename FieldTraits<T>::arg_type;
} && std::is_same_v<decltype(FieldTraits<T>::one), T>;
}  // namespace autograd

#endif  // CONCEPTS_H
