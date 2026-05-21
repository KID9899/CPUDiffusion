//
// Created by iliya on 5/21/26.
//

#include <cstdint>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>
#include <cstddef>
#include <utility>
#include <algorithm>

#include "operation.h"

#pragma once

// forward declaration
class Graph;
class Tensor;

// Класс тензора, привязанный к памяти
class UniqueTensor {
    friend class Graph;
    friend class Tensor;
private:
    Graph *graph;
    void *start;
    uintptr_t page_start;
    size_t span;
    std::vector<uint64_t> shape_;

    inline void set_memory(void *start, void *end) {
        this->start = start;
        auto page_size = sysconf(_SC_PAGESIZE);
        page_start = reinterpret_cast<uintptr_t>(start) & ~(page_size - 1);

        uintptr_t end_addr = reinterpret_cast<uintptr_t>(end);
        ptrdiff_t diff = static_cast<ptrdiff_t>(end_addr - page_start);

        if (!std::cmp_greater_equal(diff, 0)) throw std::out_of_range("The pointer to the beginning of the tensor data must be smaller than the pointer to the end");
        if (!std::cmp_less_equal(diff, SIZE_MAX)) throw std::overflow_error("The tensor data size overflows the size_t type");

        span = static_cast<size_t>(diff);
    }
public:
    inline UniqueTensor(Graph *graph, const std::vector<uint64_t>& shape): graph(graph), start(nullptr), shape_(shape) {}

    inline void prefetch() const {
        if (start == nullptr) throw std::runtime_error("The memory for the promised tensor was not allocated at the time of use");
        madvise(reinterpret_cast<void*>(page_start), span, MADV_WILLNEED);
    }
    inline const std::vector<uint64_t> &shape() const { return shape_; }
    inline void *data() const { return start; }
    Tensor reference();
};

// Класс, хранящий результат операции, непривязанный к тензору
class TensorResult {
    friend class Tensor;
private:
    OperationId id;
    const UniqueTensor *src1;
    GraphOperation::SecondArg src2;
    std::vector<uint64_t> shape;

    TensorResult(OperationId id, const UniqueTensor *src1, GraphOperation::SecondArg src2, const std::vector<uint64_t>& shape);
public:
    // Мгновенная привязка в тензору
    Tensor &operator[](Tensor &t);
};

// Безопасная обёртка над уникальным тензором
class Tensor {
    friend class UniqueTensor;
    friend class TensorResult;
    friend class Graph;
private:
    UniqueTensor *unique;
    Tensor(UniqueTensor *unique);;

    Graph *graph();;
public:
    inline Tensor(const Tensor& other): unique(other.unique) {}
    inline Tensor(Tensor&& other): unique(other.unique) { other.unique = nullptr; }
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) = default;

    inline bool promised() { return data() == nullptr; }

    inline Tensor(): unique(nullptr) {};
    void *data() const;
    const std::vector<uint64_t> &shape() const;

    // Заполнение тензора значениями
    void fill(float v);
    void randn();

    // Копирование тензора
    TensorResult copy() const;
    TensorResult transpose() const;

    // Привязка тензора к результату
    Tensor &operator=(TensorResult res);

    // Сумма-разность
    TensorResult operator+(Tensor other) const;
    TensorResult operator-(Tensor other) const;
    TensorResult operator-() const;

    // Умножение-деление
    TensorResult operator*(Tensor other) const;
    TensorResult operator*(float other) const;
    TensorResult operator/(Tensor other) const;
    TensorResult operator/(float other) const;
    TensorResult invert() const;

    // Матричное умножение
    TensorResult operator^(Tensor other) const;

    // Продвинутые операции
    TensorResult exp() const;
    TensorResult relu() const;
    TensorResult sigmoid() const;
};
