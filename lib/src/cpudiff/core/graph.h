//
// Created by iliya on 5/21/26.
//

#include <vector>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <memory>

#include "cpudiff/typeutils.h"
#include "tensor.h"
#include "operation.h"

#pragma once

// forward declaration
class SafeTensorsFile;

class Graph {
    friend class Tensor;
private:
    // Владеем всеми тензорами, кроме графа никто ими владеть не должен
    std::vector<std::unique_ptr<UniqueTensor>> tensors;
    // Владеем памятью под тензора по той же логике
    std::vector<void*> maped;
    std::vector<GraphOperation> operations;

public:
    inline Graph() = default;
    ~Graph();

    // Добавляем тензор из существующей памяти
    Tensor add_existing(void *start, void *end, const std::vector<uint64_t> &shape, Dtype dtype = Dtype::F32);

    // Создаём новый тензор
    Tensor allocate(const std::vector<uint64_t> &shape);

    // Обещаем аллокацию подходящего тензора при первой записи
    Tensor future();

    // Аллокация обещанного тензора
    void alloc_promised(UniqueTensor* tensor, const std::vector<uint64_t> &shape);

    void add_operation(OperationId id, const UniqueTensor *src1, GraphOperation::SecondArg src2, UniqueTensor *result);

    void dump_dot(const std::string &filename, const SafeTensorsFile *safetensors = nullptr) const;
};
