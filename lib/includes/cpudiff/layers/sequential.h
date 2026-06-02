//
// Created by iliya on 5/22/26.
//

#include <memory>

#include "module.h"

#pragma once

class ModuleClass(Sequential) {
public:
    std::vector<std::shared_ptr<Module>> layers;
    inline Sequential(const std::vector<std::shared_ptr<Module>> &list): layers(list) {
        for (size_t i = 0; i < layers.size(); ++i) {
            register_module(std::to_string(i), layers[i].get());
        }
    }

    AnyTensor forward(const AnyTensor &x);
};

