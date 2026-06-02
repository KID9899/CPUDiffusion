//
// Created by iliya on 5/27/26.
//

#include "module.h"

#pragma once


class ModuleClass(Embedding) {
private:
    Tensor weight;
public:
    Embedding(size_t num_embeddings, size_t embedding_dim);
    explicit Embedding(const char *weight_name = "weight");

    Tensor forward(const Tensor &x);
};

class ModuleClass(LayerNorm) {
private:
    Tensor weight, bias;
    float eps{};
public:
    LayerNorm(size_t normalized_shape, float eps = 1e-12);
    LayerNorm(const char *weight_name = "weight", const char *bias_name = "bias");

    Tensor forward(const Tensor &x);
};
