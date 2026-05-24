//
// Created by iliya on 5/22/26.
//

#include "sequential.h"

Tensor::CanAssign Sequential::forward(const Tensor &x) {
    if (layers.empty()) return x.copy();
    Tensor::CanAssign tmp = (*layers[0])(x);
    for (size_t i = 1; i < layers.size(); ++i) {
        tmp = (*layers[i])(tmp);
    }
    return tmp;
}
