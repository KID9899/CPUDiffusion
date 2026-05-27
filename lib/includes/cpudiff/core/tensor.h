//
// Created by iliya on 5/21/26.
//

#include <vector>

#include "operation.h"

#pragma once

// forward declaration
class Graph;

class Tensor {
    friend class Graph;
private:
    inline static constinit unsigned int _TENSOR_ID = 0;

    mutable Graph *graph;
    unsigned int id;
    std::vector<size_t> shape_;
    mutable std::string name_;

    Tensor(Graph *graph, const std::vector<size_t> &shape, const std::string &name = "tensor");
public:
    Tensor(): graph(nullptr), id(0), shape_(), name_("") {}
    // Установка нового имени для тензора
    const Tensor &operator[](const std::string &name) const;
    inline Tensor(const Tensor &t): graph(t.graph), id(t.id), shape_(t.shape_), name_(t.name_) {}
    inline Tensor& operator=(const Tensor& t) {
        graph = t.graph;
        id = t.id;
        shape_ = t.shape_;
        name_ = t.name_;
        return *this;
    }

    inline const std::vector<size_t> &shape() const { return shape_; }
    inline const std::string &name() const { return name_; }

    float* bind() const;

    // Поэлементные операции с тензорами
    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(const Tensor &other) const;
    Tensor operator/(const Tensor &other) const;
    // Матричное умножение
    Tensor operator^(const Tensor &other) const;

    // Поэлементные операции с float
    Tensor operator+(float val) const;
    Tensor operator-(float val) const;
    Tensor operator*(float val) const;
    Tensor operator/(float val) const;

    // Унарный минус
    Tensor operator-() const;
    // Поэлементная инверсия (1/x)
    Tensor invert() const;

    // Функции активации
    Tensor exp() const;
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;

    // Возведение в степень
    Tensor pow(float power) const;

    // Транспонирование
    Tensor transpose(int64_t dim0 = -2, int64_t dim1 = -1) const;

    // Заполнение
    void fill(float value);
    void randn();

    // Создание похожих тензоров
    Tensor like() const;
    inline static Tensor fill_like(const Tensor &s, float value) {
        Tensor r = s.like();
        r.fill(value);
        return r;
    }
    inline static Tensor randn_like(const Tensor &s) {
        Tensor r = s.like();
        r.randn();
        return r;
    }

    // Запись состояния в файл
    void repr(const char* filename) const;
    void dump(const char* filename) const;

    // Реструктурирование
    Tensor view(const std::vector<size_t> &shape) const;
    Tensor flatten() const;
};

