//
// Created by iliya on 5/22/26.
//

#include "module.h"

#pragma once

class Sequential final: public Module {
public:
    std::vector<std::shared_ptr<Module>> layers;
    Sequential(Graph *graph, std::initializer_list<std::shared_ptr<Module>> list): Module(graph), layers(list) {
        for (size_t i = 0; i < layers.size(); ++i) {
            register_module(std::to_string(i), layers[i].get());
        }
    }
protected:
    Tensor::CanAssign forward(const Tensor &x) override;
};

