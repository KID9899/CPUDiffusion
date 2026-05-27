//
// Created by iliya on 5/21/26.
//

#include "cpudiff/safetensors.h"

#include "module.h"


void Module::load(const SafeTensorsFile *file) {
    if (!file) throw std::invalid_argument("SafeTensorsFile pointer is null");
    load(file, "");
}

void Module::load(const SafeTensorsFile *file, const std::string &prefix) {
    // Загружаем веса текущего модуля
    for (auto &[name, tensor_ptr] : weights) {
        std::string full_name = prefix.empty() ? name : prefix + "." + name;
        *tensor_ptr = file->tensor(full_name);
    }

    // Рекурсивно для дочерних модулей
    for (auto &[name, child_ptr] : children) {
        std::string child_prefix = prefix.empty() ? name : prefix + "." + name;
        child_ptr->load(file, child_prefix);
    }

    on_load();  // пользовательский хук
}
