//
// Created by iliya on 5/21/26.
//

#include <vector>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <memory>
#include <unordered_set>

#include "cpudiff/typeutils.h"
#include "tensor.h"
#include "operation.h"

#pragma once

// forward declaration
class SafeTensorsFile;
class GraphExecutor;

class Graph {
    friend class Tensor;
    friend class GraphExecutor;
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
    Tensor add_existing(void *start, void *end, const std::vector<size_t> &shape, const std::string &name = "", Dtype dtype = Dtype::F32);

    // Создаём новый тензор
    Tensor allocate(const std::vector<size_t> &shape, const std::string &name = "");

    // Обещаем аллокацию подходящего тензора при первой записи
    Tensor future(const std::string &name = "");

    // Аллокация обещанного тензора
    void alloc_promised(UniqueTensor* tensor, const std::vector<size_t> &shape);

    // Добавить сырую операцию
    void add_operation(OperationId id, const UniqueTensor *src1, GraphOperation::SecondArg src2, UniqueTensor *result);

    // Сохранить граф в файл вычислений
    void dump_dot(const std::string &filename, const std::string &name = "Courier New") const;

    // Создание тензора из массива float
    Tensor link(float *data, const std::vector<size_t> &shape, const std::string& name = "");

    // Загрузка тензора из файла, записанного через Tensor::dump()
    Tensor load(const std::string& filename);
};

class GraphExecutor {
protected:
    Graph *graph;
    // Возвращает ссылку на статический список поддерживаемых переменных
    virtual const std::unordered_set<OperationId> &getSupportedOperation() const = 0;
    inline const std::vector<GraphOperation> &operations() const { return graph->operations; };
public:
    inline GraphExecutor(Graph *graph): graph(graph) {}
    virtual bool can_execute() const {
        const std::unordered_set<OperationId> &x = getSupportedOperation();
        for (const auto &i: operations()) {
            if (!x.contains(i.id)) return false;
        }
        return true;
    }
    virtual void execute() const = 0;
};
