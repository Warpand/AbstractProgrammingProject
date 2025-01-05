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
    { x* y };
    { x - y };
    { x / y };
    { x += y };
    { FieldTraits<T>::one };
    typename FieldTraits<T>::arg_type;
};
}  // namespace autograd

#endif  // CONCEPTS_H
