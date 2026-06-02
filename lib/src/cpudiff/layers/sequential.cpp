//
// Created by iliya on 5/22/26.
//

#include "sequential.h"

AnyTensor Sequential::forward(const AnyTensor &x) {
    if (layers.empty()) return x;
    AnyTensor tmp = layers[0]->operator()(x);
    for (size_t i = 1; i < layers.size(); ++i) {
        tmp = layers[i]->operator()(tmp);
    }
    return tmp;
}
