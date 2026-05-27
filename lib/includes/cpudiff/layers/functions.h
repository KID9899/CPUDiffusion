//
// Created by iliya on 5/22/26.
//

#include "module.h"

#pragma once


class ReLU final: public Module {
public:
    inline ReLU(Graph *graph): Module(graph) {}
    inline ReLU(): ReLU(Graph::get_active()) {}
protected:
    Tensor forward(const Tensor &x) override;
};

class Sigmoid final: public Module {
public:
    inline Sigmoid(Graph *graph): Module(graph) {}
    inline Sigmoid(): Sigmoid(Graph::get_active()) {}
protected:
    Tensor forward(const Tensor &x) override;
};

class Flatten final: public Module {
public:
    inline Flatten(Graph *graph): Module(graph) {}
    inline Flatten(): Flatten(Graph::get_active()) {}
protected:
    Tensor forward(const Tensor &x) override;
};

class Tanh final : public Module {
public:
    inline Tanh(Graph *graph) : Module(graph) {}
    inline Tanh(): Tanh(Graph::get_active()) {}
protected:
    Tensor forward(const Tensor &x) override;
};

//class GELU final : public Module {
//public:
//    inline GELU(Graph *graph) : Module(graph), x3(future()) {}
//protected:
//    Tensor::CanAssign forward(const Tensor &x) override;
//};
