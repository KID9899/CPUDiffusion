//
// Created by iliya on 5/22/26.
//

#include <optional>
#include <sstream>

#include "graph.h"
#include "tensor.h"

// Проверка на возможность поэлеметных операций с данными тензорами
static bool can_broadcast_shapes(const std::vector<size_t> &a, const std::vector<size_t> &b) {
    return std::equal(a.end() - b.size(), a.end(), b.begin());
}

// Вычисление размерности матричного умножения (если оно возможно)
static std::optional<std::vector<size_t>> matmul_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
    // Работаем с копиями, так как первый тензор может быть изменён
    std::vector<size_t> s1 = a;
    std::vector<size_t> s2 = b;

    // Если первый тензор одномерный, добавляем ему первое измерение 1
    if (s1.size() == 1) {
        s1.insert(s1.begin(), 1);
    }

    // Если второй тензор одномерный
    if (s2.size() == 1) {
        if (s1.empty() || s1.back() != s2[0]) {
            return std::nullopt;
        }
        return std::vector<size_t>(s1.begin(), s1.end() - 1);
    }

    // Оба тензора имеют размерность не менее 2
    if (s1.back() != s2[s2.size() - 2]) {
        return std::nullopt;
    }

    std::vector<size_t> s1_rest(s1.begin(), s1.end() - 2);
    std::vector<size_t> s2_rest(s2.begin(), s2.end() - 2);

    if (s2_rest.size() > s1_rest.size()) {
        return std::nullopt;
    }
    if (!std::equal(s1_rest.end() - s2_rest.size(), s1_rest.end(),
                    s2_rest.begin())) {
        return std::nullopt;
    }

    std::vector<size_t> result = s1_rest;
    result.push_back(s1[s1.size() - 2]);
    result.push_back(s2.back());
    return result;
}

static std::string shape_to_string(const std::vector<size_t>& s) {
    std::stringstream ss;
    ss << "[";
    if (!s.empty()) ss << s[0];
    for (size_t i = 1; i < s.size(); ++i) {
        ss << ", " << s[i];
    }
    ss << "]";
    return ss.str();
}

// Вычисление формы после транспонирования (перестановки двух осей)
static std::optional<std::vector<size_t>> transpose_shape(
        const std::vector<size_t>& shape,
        int64_t dim0 = -2,
        int64_t dim1 = -1
) {
    int64_t ndim = static_cast<size_t>(shape.size());

    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;

    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        return std::nullopt;
    }

    std::vector<size_t> new_shape = shape;
    std::swap(new_shape[dim0], new_shape[dim1]);

    return new_shape;
}

// ============= БАЗОВЫЕ ФУНКЦИИ ТЕНЗОРОВ =============

float *Tensor::bind() const {
    return graph->force_bind(*this, true);
}


// =============  АРИФМЕТИКА С ТЕНЗОРАМИ =============
Tensor Tensor::operator+(const Tensor &other) const {
    if (!id || !other.id) throw std::runtime_error("Empty tensor");
    if (graph != other.graph) throw std::runtime_error("Different graphs");
    if (!can_broadcast_shapes(shape_, other.shape_))
        throw std::runtime_error("Incompatible shapes for ADD: " + shape_to_string(shape_) + " and " + shape_to_string(other.shape_));
    return graph->add_operation(OperationId::ADD, *this, other, shape_);
}

Tensor Tensor::operator-(const Tensor &other) const {
    if (!id || !other.id) throw std::runtime_error("Empty tensor");
    if (graph != other.graph) throw std::runtime_error("Different graphs");
    if (!can_broadcast_shapes(shape_, other.shape_))
        throw std::runtime_error("Incompatible shapes for SUB: " + shape_to_string(shape_) + " and " + shape_to_string(other.shape_));
    return graph->add_operation(OperationId::SUB, *this, other, shape_);
}

Tensor Tensor::operator*(const Tensor &other) const {
    if (!id || !other.id) throw std::runtime_error("Empty tensor");
    if (graph != other.graph) throw std::runtime_error("Different graphs");
    if (!can_broadcast_shapes(shape_, other.shape_))
        throw std::runtime_error("Incompatible shapes for MUL: " + shape_to_string(shape_) + " and " + shape_to_string(other.shape_));
    return graph->add_operation(OperationId::MUL, *this, other, shape_);
}

Tensor Tensor::operator/(const Tensor &other) const {
    if (!id || !other.id) throw std::runtime_error("Empty tensor");
    if (graph != other.graph) throw std::runtime_error("Different graphs");
    if (!can_broadcast_shapes(shape_, other.shape_))
        throw std::runtime_error("Incompatible shapes for DIV: " + shape_to_string(shape_) + " and " + shape_to_string(other.shape_));
    return graph->add_operation(OperationId::DIV, *this, other, shape_);
}

Tensor Tensor::operator^(const Tensor &other) const {
    if (!id || !other.id) throw std::runtime_error("Empty tensor");
    if (graph != other.graph) throw std::runtime_error("Different graphs");
    auto opt_shape = matmul_shape(shape_, other.shape_);
    if (!opt_shape)
        throw std::runtime_error("Incompatible shapes for MATMUL: " + shape_to_string(shape_) + " and " + shape_to_string(other.shape_));
    return graph->add_operation(OperationId::MATMUL, *this, other, *opt_shape);
}

// ============= АРИФМЕТИКА С FLOAT =============

Tensor Tensor::operator+(float val) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::FADD, *this, val, shape_);
}

Tensor Tensor::operator-(float val) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::FSUB, *this, val, shape_);
}

Tensor Tensor::operator*(float val) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::FMUL, *this, val, shape_);
}

Tensor Tensor::operator/(float val) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::FDIV, *this, val, shape_);
}

// Свободные функции для float слева (только коммутативные)
Tensor operator+(float val, const Tensor &t) { return t + val; }
Tensor operator-(float val, const Tensor &t) {
    if (std::abs(val) < 1e-5f) return -t;
    return val + (-t);
}
Tensor operator*(float val, const Tensor &t) { return t * val; }
Tensor operator/(float val, const Tensor &t) {
    if (std::abs(val - 1) < 1e-5f) return t.invert();
    return val * t.invert();
}

Tensor Tensor::pow(float val) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::POW, *this, val, shape_);
}

// ============= УНАРНЫЕ ОПЕРАЦИИ =============

Tensor Tensor::operator-() const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::NEGATE, *this, nullptr, shape_);
}

Tensor Tensor::invert() const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::INVERT, *this, nullptr, shape_);
}

Tensor Tensor::exp() const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::EXP, *this, nullptr, shape_);
}

Tensor Tensor::relu() const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::RELU, *this, nullptr, shape_);
}

Tensor Tensor::sigmoid() const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::SIGMOID, *this, nullptr, shape_);
}

Tensor Tensor::tanh() const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::TANH, *this, nullptr, shape_);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    if (!id) throw std::runtime_error("Empty tensor");
    auto opt_shape = transpose_shape(shape_, dim0, dim1);
    if (!opt_shape)
        throw std::runtime_error("Invalid transpose dimensions");
    return graph->add_operation(OperationId::TRANSPOSE, *this, nullptr, *opt_shape);
}

// ============= ОПЕРАЦИИ НА МЕСТЕ =============

void Tensor::fill(float value) {
    if (!id) throw std::runtime_error("Empty tensor");
    graph->add_operation(OperationId::FILL, *this, value, shape_);
}

void Tensor::randn() {
    if (!id) throw std::runtime_error("Empty tensor");
    graph->add_operation(OperationId::RANDN, *this, nullptr, shape_);
}

void Tensor::repr(const char* filename) const {
    if (!id) throw std::runtime_error("Empty tensor");
    graph->add_operation(OperationId::REPR, *this, filename, {});
}

void Tensor::dump(const char* filename) const {
    if (!id) throw std::runtime_error("Empty tensor");
    graph->add_operation(OperationId::DUMP, *this, filename, {});
}

// ============= ОПЕРАЦИИ СОЗДАНИЯ ПОХОЖИХ =============

Tensor Tensor::like() const {
    return graph->add_operation(OperationId::LIKE, *this, nullptr, shape_);
}


// ============= ОПЕРАЦИИ РЕСТРУКТУРИРОВАНИЯ =============

Tensor Tensor::view(const std::vector<size_t> &shape) const {
    if (!id) throw std::runtime_error("Empty tensor");
    size_t total_elements = 1;
    for (size_t d : shape_) total_elements *= d;

    // Ищем ось для автоматического вывода
    size_t known_product = 1;
    size_t infer_index = 0;
    size_t infer_count = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) {
            infer_index = i;
            ++infer_count;
        } else {
            known_product *= shape[i];
        }
    }

    std::vector<size_t> new_shape = shape;
    if (infer_count == 0) {
        // Проверяем точное совпадение числа элементов
        size_t new_total = 1;
        for (size_t d : new_shape) new_total *= d;
        if (total_elements != new_total) {
            throw std::runtime_error(
                    "Cannot reshape tensor: total elements mismatch. "
                    "Original: " + std::to_string(total_elements) +
                    ", requested: " + std::to_string(new_total));
        }
    } else if (infer_count == 1) {
        // Автоматически выводим одну размерность
        if (known_product == 0 || total_elements % known_product != 0) {
            throw std::runtime_error(
                    "Cannot infer dimension: total elements " + std::to_string(total_elements) +
                    " not divisible by product of known dimensions " + std::to_string(known_product));
        }
        new_shape[infer_index] = total_elements / known_product;
    } else {
        throw std::runtime_error("Only one dimension can be inferred (multiple 0 found)");
    }

    // Создаём операцию в графе
    return graph->add_operation(OperationId::VIEW, *this, nullptr, new_shape);
}
Tensor Tensor::flatten() const {
    size_t second = 1;
    const std::vector<size_t> &s = shape();
    for (size_t i = 1; i < s.size(); ++i) second *= s[i];
    return view({0, second});
}

Tensor::Tensor(Graph *graph, const std::vector<size_t> &shape, const std::string &name) : graph(graph), id(++_TENSOR_ID), shape_(shape), name_(name) {
    graph->names[id] = name;
}
const Tensor &Tensor::operator[](const std::string &name) const {
    if (!id) throw std::runtime_error("Empty tensor");
    name_ = name;
    graph->names[id] = name;
    return *this;
}

