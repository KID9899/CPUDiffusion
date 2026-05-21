//
// Created by iliya on 5/22/26.
//

#include "activators.h"

TensorResult ReLU::forward(const Tensor &x) {
    return x.relu();
}

TensorResult Sigmoid::forward(const Tensor &x) {
    return x.sigmoid();
}

TensorResult Flatten::forward(const Tensor &x) {
    return x.copy();
}
