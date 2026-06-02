//
// Created by iliya on 5/31/26.
//

#include "module.h"

#pragma once

class ModuleClass(ModuleList) {
private:
    std::vector<std::shared_ptr<Module>> layers;
public:
    inline ModuleList(std::vector<std::shared_ptr<Module>> list): layers(list) {
        for (size_t i = 0; i < layers.size(); ++i) {
            register_module(std::to_string(i), layers[i].get());
        }
    }

    inline Module &operator[](size_t idx) { return get(idx); }
    inline Module &get(size_t idx) { return *layers[idx]; }
    inline size_t size() { return layers.size(); }
    Tensor forward() { throw std::logic_error("ModuleList is not callable"); };
};
