//
// Created by iliya on 5/22/26.
//

#include "module.h"

#pragma once

class ModuleClass(Linear) {
private:
    size_t in_size{}, out_size{};

public:
    Tensor weight;
    Tensor bias;

    Linear(size_t in_size, size_t out_size);
    Linear(const char *weight_name = "weight", const char *bias_name = "bias");

    Tensor forward(const Tensor &x) const;
};

class ModuleClass(LinearMatrix) {
private:
    size_t in_size{}, out_size{};

public:
    Tensor weight;

    LinearMatrix(size_t in_size, size_t out_size);
    LinearMatrix(const char *weight_name = "weight");

    Tensor forward(const Tensor &x) const;
};