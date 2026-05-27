//
// Created by iliya on 5/22/26.
//

#include "linear.h"

Linear::Linear(Graph *graph, size_t in_size, size_t out_size): Module(graph), in_size(in_size), out_size(out_size) {
    register_tensor("weight", weight);
    register_tensor("bias", bias);
}
Linear::Linear(size_t in_size, size_t out_size): Linear(Graph::get_active(), in_size, out_size) {}

Tensor Linear::forward(const Tensor &x) {
    return (x ^ weight.transpose()) + bias;
}
