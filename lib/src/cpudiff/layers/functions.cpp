//
// Created by iliya on 5/22/26.
//

#include "functions.h"

Tensor ReLU::forward(const Tensor &x) {
    return x.relu();
}

Tensor Sigmoid::forward(const Tensor &x) {
    return x.sigmoid();
}

Tensor Flatten::forward(const Tensor &x) {
    return x.flatten();
}

Tensor Tanh::forward(const Tensor &x) {
    return x.tanh();
}

//Tensor GELU::forward(const Tensor &x) {
//    // GELU(x) ~ 0.5 * x * (1 + tanh(√(2/pi) * (x + 0.044715 * x^3)))
//    static const float sqrt_2_pi = 0.7978845608028654f;
//    static const float coeff = 0.044715f;
//    x3 = x.pow(3.0f);
//    x3 = x3 * coeff;
//    x3 += x;
//    x3 *= sqrt_2_pi;
//    x3 = x3.tanh();
//    x3 += 1.0f;
//    x3 *= 0.5f;
//    return x * x3;
//}
