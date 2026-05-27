//
// Created by iliya on 5/22/26.
//

#include "module.h"

#pragma once

class Linear final: public Module {
private:
    size_t in_size, out_size;

public:
    Tensor weight;
    Tensor bias;

    Linear(Graph *graph, size_t in_size, size_t out_size);
    Linear(size_t in_size, size_t out_size);
protected:
    Tensor forward(const Tensor &x) override;
};
