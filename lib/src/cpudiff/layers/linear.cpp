//
// Created by iliya on 5/22/26.
//

#include "linear.h"

Linear::Linear(size_t in_size, size_t out_size): in_size(in_size), out_size(out_size) {
    weight = graph->allocate({out_size, in_size}, "weight");
    bias   = graph->allocate({out_size}, "bias");
}

Linear::Linear(const char *weight_name, const char *bias_name) {
    register_tensor(weight_name, weight);
    register_tensor(bias_name, bias);
}

Tensor Linear::forward(const Tensor &x) const {
    return (x ^ weight.transpose()) + bias;
}

LinearMatrix::LinearMatrix(size_t in_size, size_t out_size): in_size(in_size), out_size(out_size) {
    weight = graph->allocate({out_size, in_size}, "weight");
}

LinearMatrix::LinearMatrix(const char *weight_name) {
    register_tensor(weight_name, weight);
}

Tensor LinearMatrix::forward(const Tensor &x) const {
    return x ^ weight.transpose();
}
