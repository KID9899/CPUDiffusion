//
// Created by iliya on 5/21/26.
//

#include <unordered_map>
#include <stdexcept>

#include "cpudiff/core.h"
#include "cpudiff/safetensors.h"

#pragma once

class Module {
private:
    std::unordered_map<std::string, Module*> children;
    std::unordered_map<std::string, Tensor*> weights;
    bool finalized;
    void load(const SafeTensorsFile *file, const std::string &prefix);
protected:
    Graph *graph;
    Tensor input;
    // Основная функция для переопределения поведения модуля
    virtual Tensor::CanAssign forward(const Tensor &x) = 0;
    // Дополнительная функция, выполняющаяся после загрузки тензора
    virtual void on_load() {};

    inline void register_module(const std::string &name, Module &module) {
        if (children.contains(name) || weights.contains(name)) throw std::logic_error("Two descendants of a module cannot have the same name");
        children[name] = &module;
    }
    inline void register_module(const std::string &name, Module *module) {
        if (children.contains(name) || weights.contains(name)) throw std::logic_error("Two descendants of a module cannot have the same name");
        children[name] = module;
    }
    inline void register_tensor(const std::string &name, Tensor &tensor) {
        if (children.contains(name) || weights.contains(name)) throw std::logic_error("Two descendants of a module cannot have the same name");
        weights[name] = &tensor;
    }
    inline Tensor future() {
        return graph->future();
    }
    inline Tensor::CanAssign forward(const Tensor::CanAssign &x) {
        input = x;
        return forward(input);
    };
public:
    inline Module(Graph *graph): graph(graph) {
        input = future();
    }
    void load(const SafeTensorsFile *file);
    inline Tensor::CanAssign operator()(const Tensor &x) { return forward(x); }
    inline Tensor::CanAssign operator()(const Tensor::CanAssign &x) { return forward(x); }
};
