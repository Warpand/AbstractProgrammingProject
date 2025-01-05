#ifndef CONTEXT_H
#define CONTEXT_H

namespace autograd {
template <typename T>
class GradContext {
    static int context_counter;

    GradContext() { context_counter++; }

   public:
    [[nodiscard]] static bool grad_enabled() { return context_counter == 0; }

    static GradContext no_grad() { return GradContext(); }

    ~GradContext() { context_counter--; }
};

template <typename T>
int GradContext<T>::context_counter = 0;
}  // namespace autograd

#endif  // CONTEXT_H
