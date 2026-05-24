//
// Created by iliya on 5/22/26.
//

#include "functions.h"

Tensor::CanAssign ReLU::forward(const Tensor &x) {
    return x.relu();
}

Tensor::CanAssign Sigmoid::forward(const Tensor &x) {
    return x.sigmoid();
}

Tensor::CanAssign Flatten::forward(const Tensor &x) {
    return x.flatten();
}
