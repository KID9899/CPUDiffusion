//
// Created by iliya on 5/20/26.
//

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

#include "cpudiff/typeutils.h"
#include "cpudiff/core.h"

#pragma once

class SafeTensorsFile {
public:
    SafeTensorsFile(const std::string &path, Graph *graph);
    inline SafeTensorsFile(const std::string &path): SafeTensorsFile(path, Graph::get_active()) {};

    inline const std::vector<std::string> &names() const { return names_; }
    inline const std::vector<Tensor> &tensors() const { return tensors_; }
    const Tensor &tensor(const std::string &name) const {
        auto it = name_to_idx_.find(name);
        if (it == name_to_idx_.end())
            throw std::out_of_range("Tensor not found: " + name);
        return tensors_[it->second];
    }

private:
    Graph *graph;
    std::vector<std::string> names_;
    std::vector<Tensor> tensors_;
    std::unordered_map<std::string, size_t> name_to_idx_;
};