//
// Created by iliya on 5/27/26.
//

#include "transformer_parts.h"

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim) {
    weight = graph->allocate({num_embeddings, embedding_dim}, "weight");
}

Embedding::Embedding(const char *weight_name) {
    register_tensor(weight_name, weight);
}


Tensor Embedding::forward(const Tensor &x) {
    return weight.gather(x.as_index(0));
}


LayerNorm::LayerNorm(size_t normalized_shape, float eps): eps(eps) {
    weight = graph->allocate({normalized_shape}, "weight");
    bias   = graph->allocate({normalized_shape}, "bias");
    weight.fill(1.0f);
    bias.fill(0.0f);
}

LayerNorm::LayerNorm(const char *weight_name, const char *bias_name) {
    register_tensor(weight_name, weight);
    register_tensor(bias_name, bias);
}


Tensor LayerNorm::forward(const Tensor &x) {
    Tensor mean = x.reduce_mean(-1).repeat(x.shape().back());
    Tensor diff = x - mean;
    Tensor var = (diff * diff).reduce_mean(-1).repeat(x.shape().back());
    Tensor normed = diff / (var + eps).pow(0.5f);
    return normed * weight + bias;
}
