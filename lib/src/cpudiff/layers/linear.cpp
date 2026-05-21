//
// Created by iliya on 5/22/26.
//

#include "linear.h"

Linear::Linear(Graph *graph, uint64_t in_size, uint64_t out_size): Module(graph), in_size(in_size), out_size(out_size) {
    matmul_result = future();
    weight_t = future();

    register_tensor("weight", weight);
    register_tensor("bias", bias);
}

TensorResult Linear::forward(const Tensor &x) {
    matmul_result = x ^ weight_t;
    return matmul_result + bias;
}

void Linear::on_load() {
    weight_t = weight.transpose();
}
