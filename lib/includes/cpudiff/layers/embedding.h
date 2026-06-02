//
// Created by iliya on 5/27/26.
//

#include "module.h"

#pragma once


class Embedding final: public Module {
    Tensor weight;
public:
    inline Embedding(Graph *graph): Module(graph) { register_tensor("weight", weight); }
    inline Embedding(): Embedding(Graph::get_active()) {}
protected:
    Tensor forward(const Tensor &x) override;
};
