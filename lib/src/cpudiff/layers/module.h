//
// Created by iliya on 5/21/26.
//

#include <unordered_map>
#include <stdexcept>
#include <string>

#include "cpudiff/core.h"
#include "cpudiff/safetensors.h"

#pragma once

class Module {
private:
    std::unordered_map<std::string, Module*> children;
    std::unordered_map<std::string, Tensor*> weights;
    void load(const SafeTensorsFile *file, const std::string &prefix);

protected:
    Graph *graph;

    // Основной метод, который нужно переопределить
    virtual Tensor forward(const Tensor &x) = 0;

    // Вызывается после загрузки весов, можно переопределить
    virtual void on_load() {}

    // Регистрация подмодулей и параметров
    void register_module(const std::string &name, Module &module) {
        if (children.contains(name) || weights.contains(name))
            throw std::logic_error("Two descendants of a module cannot have the same name");
        children[name] = &module;
    }
    void register_module(const std::string &name, Module *module) {
        if (children.contains(name) || weights.contains(name))
            throw std::logic_error("Two descendants of a module cannot have the same name");
        children[name] = module;
    }
    void register_tensor(const std::string &name, Tensor &tensor) {
        if (children.contains(name) || weights.contains(name))
            throw std::logic_error("Two descendants of a module cannot have the same name");
        weights[name] = &tensor;
    }

public:
    explicit Module(Graph *graph): graph(graph) {}
    explicit Module(): graph(Graph::get_active()) {}

    // Загрузить веса и рекурсивно подмодули
    void load(const SafeTensorsFile *file);

    // Вызов модуля как функции - единственный способ применить модуль извне
    virtual Tensor operator()(const Tensor &x) {
        return forward(x);
    }
};
