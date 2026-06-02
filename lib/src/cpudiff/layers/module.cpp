//
// Created by iliya on 5/21/26.
//

#include "cpudiff/safetensors.h"

#include "module.h"

void Module::load(const SafeTensorsFile *file) {
    if (!file) throw std::invalid_argument("SafeTensorsFile pointer is null");
    load(file, "");
}

// NOLINTNEXTLINE(misc-no-recursion)
void Module::load(const SafeTensorsFile *file, const std::string &prefix) {
    // Загружаем веса текущего модуля
    for (auto &[name, tensor_ptr] : weights) {
        std::string full_name = prefix + ".";
        full_name += name;
        if (tensor_ptr->get_id()) throw std::runtime_error("Tensor " + tensor_ptr->name() + " is already bound");
        *tensor_ptr = file->tensor(prefix.empty() ? name : full_name);
    }
    for (auto &[name, index_ptr] : indexes) {
        std::string full_name = prefix + ".";
        full_name += name;
        if (index_ptr->data) throw std::runtime_error("IndexList " + full_name + " is already bound");
        *index_ptr = file->tensor(prefix.empty() ? name : full_name).as_index();
    }

    // Рекурсивно для дочерних модулей
    for (auto &[name, child_ptr] : children) {
        std::string child_prefix = prefix + ".";
        child_prefix += name;
        child_ptr->load(file, prefix.empty() ? name : child_prefix);
    }

    on_load();  // пользовательский хук
}

void Module::register_module(const std::string &name, Module &module) {
    register_module(name, &module);
}
void Module::register_module(const std::string &name, Module *module) {
    if (children.contains(name) || weights.contains(name) || indexes.contains(name))
        throw std::logic_error("Two descendants of a module cannot have the same name");
    children[name] = module;
    module->module_name = module_name + "." + name;
}
void Module::register_tensor(const std::string &name, Tensor &tensor) {
    if (children.contains(name) || weights.contains(name) || indexes.contains(name))
        throw std::logic_error("Two descendants of a module cannot have the same name");
    weights[name] = &tensor;
    if (tensor.get_id()) {
        tensor[module_name + "." + name];
    }
}
void Module::register_indexes(const std::string &name, IndexList &list) {
    if (children.contains(name) || weights.contains(name) || indexes.contains(name))
        throw std::logic_error("Two descendants of a module cannot have the same name");
    indexes[name] = &list;
}

Module::Module(Graph *graph) : graph(graph) { set_module_name(); }
