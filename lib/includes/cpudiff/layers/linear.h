//
// Created by iliya on 5/22/26.
//

#include "module.h"

#pragma once

class Linear final: public Module {
private:
    uint64_t in_size, out_size;

    Tensor matmul_result;
    Tensor weight_t;
public:
    Tensor weight;
    Tensor bias;

    Linear(Graph *graph, uint64_t in_size, uint64_t out_size);
protected:
    TensorResult forward(const Tensor &x) override;
    void on_load() override;
};
