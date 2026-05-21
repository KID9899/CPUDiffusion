//
// Created by iliya on 5/22/26.
//

#include "module.h"

#pragma once


class ReLU final: public Module {
public:
    inline ReLU(Graph *graph): Module(graph) {}
protected:
    TensorResult forward(const Tensor &x) override;
};

class Sigmoid final: public Module {
public:
    inline Sigmoid(Graph *graph): Module(graph) {}
protected:
    TensorResult forward(const Tensor &x) override;
};

class Flatten final: public Module {
public:
    inline Flatten(Graph *graph): Module(graph) {}
protected:
    TensorResult forward(const Tensor &x) override;
};
