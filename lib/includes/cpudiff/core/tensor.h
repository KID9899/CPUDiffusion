//
// Created by iliya on 5/21/26.
//

#include <vector>

#include "cpudiff/typeutils.h"
#include "operation.h"

#pragma once

// forward declaration
class Graph;

struct IndexList {
    const CompatibleInt* data = nullptr;
    size_t       size = 0;
    unsigned int axis = 0;

    IndexList() = default;
    IndexList(const CompatibleInt* d, size_t s, unsigned int a)
            : data(d), size(s), axis(a) {};
};

class Tensor {
    friend class Graph;
private:
    inline static constinit unsigned int TENSOR_ID = 0;

    mutable Graph *graph;
    unsigned int id;
    std::vector<size_t> shape_;
    std::string name_;

    Tensor(Graph *graph, const std::vector<size_t> &shape, const std::string &name = "tensor");

    // Получить указатель на результирующий тензор
    float *touch_() const;
public:
    Tensor(): graph(nullptr), id(0), shape_() {}
    // Установка нового имени для тензора
    Tensor &operator[](const std::string_view &name);
    inline Tensor(const Tensor &t) = default;
    inline Tensor& operator=(const Tensor& t) = default;

    inline const std::vector<size_t> &shape() const { return shape_; }
    inline const std::string &name() const { return name_; }
    inline unsigned int get_id() const { return id; }

    // Сделать тензор результирующим
    Tensor &bind();
    // Получить указатель на результирующий тензор
    // или субтензор по индексам
    template <typename... Args>
    requires (std::convertible_to<Args, int> && ...)
    const float *touch(Args... idx) const {
        const float *base = touch_();
        constexpr std::size_t N = sizeof...(Args);
        if constexpr (N == 0) {
            return base;
        }
        if (N > shape_.size()) {
            throw std::invalid_argument("Too many indices");
        }

        int raw_indices[] = { static_cast<int>(idx)... };
        size_t indices[N];

        for (size_t i = 0; i < N; ++i) {
            if (raw_indices[i] < 0) raw_indices[i] += shape_[i];
            if (raw_indices[i] >= shape_[i] || raw_indices[i] < 0) {
                throw std::out_of_range("Index out of bounds");
            }
            indices[i] = static_cast<size_t>(raw_indices[i]);
        }

        std::size_t offset = 0;
        std::size_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            if (i < static_cast<int>(N)) {
                offset += indices[i] * stride;
            }
            stride *= shape_[i];
        }

        std::vector<std::size_t> sub_shape(shape_.begin() + N, shape_.end());

        return base + offset;
    }

    // Поэлементные операции с тензорами
    Tensor add(const Tensor &other) const;
    Tensor sub(const Tensor &other) const;
    Tensor mul(const Tensor &other) const;
    Tensor div(const Tensor &other) const;
    // Матричное умножение
    Tensor dot(const Tensor &other) const;

    // Поэлементные операции с float
    Tensor add(float val) const;
    Tensor sub(float val) const;
    Tensor mul(float val) const;
    Tensor div(float val) const;

    // Унарный минус
    Tensor neg() const;
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
    Tensor transpose() const;

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
    void repr_now(const char* filename) const;
    void dump_now(const char* filename) const;

    // Реструктурирование
    Tensor view(const std::vector<size_t> &shape) const;
    Tensor flatten() const;

    // Преобразовать тензор в IndexList (только один раз)
    IndexList as_index(unsigned int axis = 0) const;

    // Взятие по индексам
    Tensor gather(const IndexList& indices) const;
    Tensor operator[](const IndexList& indices) const { return gather(indices); }

    // Редуцирующие операции
    Tensor reduce_min(int axis, bool savedim = true) const;
    Tensor reduce_max(int axis, bool savedim = true) const;
    Tensor reduce_sum(int axis, bool savedim = true) const;
    Tensor reduce_mean(int axis, bool savedim = true) const;

    // Расширение тензора по последней размерности
    Tensor repeat(size_t count) const;

    inline explicit operator bool() const { return id; }
};

inline Tensor operator+(const Tensor &a, const Tensor &b) { return a.add(b); }
inline Tensor operator-(const Tensor &a, const Tensor &b) { return a.sub(b); }
inline Tensor operator+(const Tensor &a, float b) { return a.add(b); }
inline Tensor operator-(const Tensor &a, float b) { return a.sub(b); }

inline Tensor operator-(const Tensor &a) { return a.neg(); }

inline Tensor operator*(const Tensor &a, const Tensor &b) { return a.mul(b); }
inline Tensor operator/(const Tensor &a, const Tensor &b) { return a.div(b); }
inline Tensor operator*(const Tensor &a, float b) { return a.mul(b); }
inline Tensor operator/(const Tensor &a, float b) { return a.div(b); }

inline Tensor operator^(const Tensor &a, const Tensor &b) { return a.dot(b); }


// Свободные функции для float слева (только коммутативные)
inline Tensor operator+(float val, const Tensor &t) { return t + val; }
inline Tensor operator-(float val, const Tensor &t) {
    if (std::abs(val) < 1e-5f) return -t;
    return (-t) + val;
}
inline Tensor operator*(float val, const Tensor &t) { return t * val; }
inline Tensor operator/(float val, const Tensor &t) {
    if (std::abs(val - 1) < 1e-5f) return t.invert();
    return t.invert() * val;
}

namespace CDMath {
    // Функции активации
    inline Tensor exp(const Tensor &t) { return t.exp(); }
    inline Tensor relu(const Tensor &t) { return t.relu(); }
    inline Tensor sigmoid(const Tensor &t) { return t.sigmoid(); }
    inline Tensor tanh(const Tensor &t) { return t.tanh(); }

    // Возведение в степень
    inline Tensor pow(const Tensor &t, float power) { return t.pow(power); }
    // Транспонирование
    inline Tensor transpose(const Tensor &t) { return t.transpose(); }

    Tensor softmax(const Tensor &t, int dim = -1);
}
