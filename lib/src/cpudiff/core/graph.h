//
// Created by iliya on 5/21/26.
//

#include <vector>
#include <algorithm>
#include <cstdint>
#include <variant>
#include <sys/mman.h>
#include <unordered_map>
#include <memory>

#include "cpudiff/typeutils.h"
#include "tensor.h"
#include "operation.h"

#pragma once

// Не связанная с памятью операция
struct IncompleteOperation {
    using ARG2_t = std::variant<unsigned int, float, const char*, IndexList>;
    OperationId id;
    unsigned int arg1;
    ARG2_t arg2;
    unsigned int result;
    std::vector<size_t> arg1_shape;
    std::vector<size_t> arg2_shape;
    std::vector<size_t> result_shape;
};

// Класс тензора, привязанный к конкретной памяти
struct BoundTensor {
    float *start, *end;
    std::vector<size_t> shape;

    inline void prefetch() const {
        if (start == nullptr) throw std::runtime_error("The memory for the promised tensor was not allocated at the time of use");
        madvise(reinterpret_cast<void*>(start), static_cast<size_t>(end - start), MADV_WILLNEED);
    }

    // Вывод представления тензора в поток
    void repr(std::ostream &out) const;
    // Бинарная запись тензора в поток
    void dump(std::ostream &out) const;
};

// Привязанная к памяти операция
struct CompleteOperation {
    using ARG2_t = std::variant<BoundTensor, float, const char*, IndexList, size_t>;
    OperationId id;
    BoundTensor arg1;
    ARG2_t arg2;
    BoundTensor result;
};

class Graph {
    friend class Tensor;
    friend class GraphExecutor;
private:
    // Указатель на текущий активный граф (изменяется при помощи Graph::with или при создании графа)
    // Любой класс, привязанный к графу, привяжется по умолчанию к активному графу
    // При удалении активного графа ставится в nullptr
    inline static thread_local Graph* current_graph = nullptr;

    // Собран ли уже граф (готов ли он к выполнению)
    bool compiled = false;

    // Владеем участками памяти, которые аллоцирует граф
    std::vector<void*> mapped;
    // Храним участки памяти, с которыми мы можем работать
    std::unordered_map<float*, size_t> usable;
    // Храним множество результирующих векторов, на данные которых мы должны будет предоставить указатель
    std::unordered_set<unsigned int> results;
    // Держим список операций на фронтенде
    std::vector<IncompleteOperation> promised;
    // Держим список будущих операций на бэкенде
    std::vector<CompleteOperation> operations;
    // Соотношение id тензора -> начальный участок в памяти для входных и выходных тензоров
    std::unordered_map<unsigned int, float*> bound;
    // Мапа с именами тензоров
    std::unordered_map<unsigned int, std::string> names;
    // Множество id тензоров, содержащих индексы
    // TODO - в длительной перспективе избежать преобразования int->float->int при загрузке файла
    std::unordered_set<unsigned int> size_tensors;

    inline Tensor add_operation(OperationId id,
                                const Tensor &arg1,
                                const std::variant<Tensor, float, const char*, IndexList, size_t> &arg2,
                                const std::vector<size_t> &out_shape)
    {
        if (compiled) throw std::runtime_error("You cannot add operation to compiled graph");
        if (size_tensors.contains(arg1.id) && !OperationGroups::arg2_is_indexlist.contains(id))
            throw std::runtime_error("It is impossible to perform tensor operations with the size tensor");

        Tensor result;
        if (!OperationGroups::no_result.contains(id)) {
            result = Tensor(this, out_shape);
        }

        if (OperationGroups::arg2_is_tensor.contains(id)) {
            auto t = std::get<Tensor>(arg2);
            if (size_tensors.contains(t.id)) throw std::runtime_error("Size tensor used in incompatible operation");
            promised.push_back({id, arg1.id, t.id, result.id, arg1.shape(), t.shape(), result.shape()});
        } else if (OperationGroups::arg2_is_float.contains(id)) {
            promised.push_back({id, arg1.id, std::get<float>(arg2), result.id, arg1.shape(), {}, result.shape()});
        } else if (OperationGroups::arg2_is_string.contains(id)) {
            promised.push_back({id, arg1.id, std::get<const char*>(arg2), result.id, arg1.shape(), {}, result.shape()});
        } else if (OperationGroups::arg2_is_null.contains(id)) {
            promised.push_back({id, arg1.id, 0u, result.id, arg1.shape(), {}, result.shape()});
        } else if (OperationGroups::arg2_is_indexlist.contains(id)) {
            const auto& idx = std::get<IndexList>(arg2);
            promised.push_back({id, arg1.id, idx, result.id, arg1.shape(), {idx.size}, result.shape()});
        } else if (OperationGroups::arg2_is_size.contains(id)) {
            size_t axis = std::get<size_t>(arg2);
            promised.push_back({id, arg1.id, static_cast<unsigned int>(axis), result.id, arg1.shape(), {}, result.shape()});
        } else {
            throw std::runtime_error(std::string("It is not known how to work with operation ") + operation_name(id));
        }

        if (OperationGroups::no_result.contains(id)) {
            return arg1;
        } else {
            return result;
        }
    }

    // Переопределение тензора как тензора индексов
    const CompatibleInt *to_size_tensor(const Tensor &t);
public:
    inline Graph() { current_graph = this; }
    inline void with() { current_graph = this; }
    inline static void with(Graph &g) { current_graph = &g; }
    inline static Graph *get_active() { return current_graph; };
    ~Graph();

    // Создаём новый тензор в памяти
    Tensor allocate(const std::vector<size_t> &shape, const std::string &name = "tensor");
    // Привязываем память к существующему тензору
    Tensor link(float *data, const std::vector<size_t> &shape, const std::string &name = "tensor");
    // Насильно аллоцируем память под тензор (если она ещё не выделена) и получаем на неё указатель
    float* force_bind(const Tensor &t, bool result = false);
    // Загрузка тензора из бинарного файла, записанного функцией dump
    Tensor load(const std::string &filename, const std::string &name = "tensor");

    void compile();

    void repr(const std::string &filename, const std::string &fontname = "Arial") const;

    void repr_compiled(const char* filename) const;
    void repr_compiled(std::ostream& os) const;

};

class GraphExecutor {
protected:
    Graph *graph;
    [[nodiscard]] virtual const std::unordered_set<OperationId> &getSupportedOperation() const = 0;
    [[nodiscard]] inline const std::vector<CompleteOperation> &operations() const { return graph->operations; };
public:
    inline explicit GraphExecutor(Graph *graph): graph(graph) {}
    inline GraphExecutor(): GraphExecutor(Graph::get_active()) {}
    [[nodiscard]] virtual bool can_execute() const {
        const std::unordered_set<OperationId> &x = getSupportedOperation();
        return std::ranges::all_of(operations(), [&x](const auto &i){ return x.contains(i.id); });
    }
    virtual void execute() const = 0;
};
