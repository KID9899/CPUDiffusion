//
// Created by iliya on 5/22/26.
//

#include <optional>
#include <fstream>
#include <sstream>

#include "graph.h"
#include "tensor.h"

// Проверка на возможность поэлементных операций с данными тензорами
static bool can_broadcast_shapes(const std::vector<size_t> &a, const std::vector<size_t> &b) {
    return std::equal(a.end() - static_cast<ptrdiff_t>(b.size()), a.end(), b.begin());
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
    if (!std::equal(s1_rest.end() - static_cast<ptrdiff_t>(s2_rest.size()), s1_rest.end(),
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
// TODO - добавить произвольное транспонирование
static std::optional<std::vector<size_t>> transpose_shape(
        const std::vector<size_t>& shape,
        int dim0 = -2,
        int dim1 = -1
) {
    int ndim = static_cast<int>(shape.size());

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

float *Tensor::touch_() const {
    if (!graph->compiled) throw std::runtime_error("It is impossible to extract a tensor from an uncompiled graph");
    return graph->force_bind(*this, true);
}
Tensor &Tensor::bind() {
    graph->force_bind(*this, true);
    return *this;
}


// =============  АРИФМЕТИКА С ТЕНЗОРАМИ =============
Tensor Tensor::add(const Tensor &other) const {
    if (!id || !other.id) throw std::runtime_error("Empty tensor");
    if (graph != other.graph) throw std::runtime_error("Different graphs");
    if (!can_broadcast_shapes(shape_, other.shape_))
        throw std::runtime_error("Incompatible shapes for ADD: " + shape_to_string(shape_) + " and " + shape_to_string(other.shape_));
    return graph->add_operation(OperationId::ADD, *this, other, shape_);
}

Tensor Tensor::sub(const Tensor &other) const {
    if (!id || !other.id) throw std::runtime_error("Empty tensor");
    if (graph != other.graph) throw std::runtime_error("Different graphs");
    if (!can_broadcast_shapes(shape_, other.shape_))
        throw std::runtime_error("Incompatible shapes for SUB: " + shape_to_string(shape_) + " and " + shape_to_string(other.shape_));
    return graph->add_operation(OperationId::SUB, *this, other, shape_);
}

Tensor Tensor::mul(const Tensor &other) const {
    if (!id || !other.id) throw std::runtime_error("Empty tensor");
    if (graph != other.graph) throw std::runtime_error("Different graphs");
    if (!can_broadcast_shapes(shape_, other.shape_))
        throw std::runtime_error("Incompatible shapes for MUL: " + shape_to_string(shape_) + " and " + shape_to_string(other.shape_));
    return graph->add_operation(OperationId::MUL, *this, other, shape_);
}

Tensor Tensor::div(const Tensor &other) const {
    if (!id || !other.id) throw std::runtime_error("Empty tensor");
    if (graph != other.graph) throw std::runtime_error("Different graphs");
    if (!can_broadcast_shapes(shape_, other.shape_))
        throw std::runtime_error("Incompatible shapes for DIV: " + shape_to_string(shape_) + " and " + shape_to_string(other.shape_));
    return graph->add_operation(OperationId::DIV, *this, other, shape_);
}

Tensor Tensor::dot(const Tensor &other) const {
    if (!id || !other.id) throw std::runtime_error("Empty tensor");
    if (graph != other.graph) throw std::runtime_error("Different graphs");
    auto opt_shape = matmul_shape(shape_, other.shape_);
    if (!opt_shape)
        throw std::runtime_error("Incompatible shapes for MATMUL: " + shape_to_string(shape_) + " and " + shape_to_string(other.shape_));
    return graph->add_operation(OperationId::MATMUL, *this, other, *opt_shape);
}

// ============= АРИФМЕТИКА С FLOAT =============

Tensor Tensor::add(float val) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::FADD, *this, val, shape_);
}

Tensor Tensor::sub(float val) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::FSUB, *this, val, shape_);
}

Tensor Tensor::mul(float val) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::FMUL, *this, val, shape_);
}

Tensor Tensor::div(float val) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::FDIV, *this, val, shape_);
}

Tensor Tensor::pow(float val) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::POW, *this, val, shape_);
}

// ============= УНАРНЫЕ ОПЕРАЦИИ =============

Tensor Tensor::neg() const {
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

Tensor Tensor::transpose() const {
    if (!id) throw std::runtime_error("Empty tensor");
    auto opt_shape = transpose_shape(shape_);
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

void Tensor::repr_now(const char* filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error(std::string("Cannot open file: ") + filename);
    size_t num_elements = 1;
    for (const auto &i: shape_) {
        num_elements *= i;
    }
    BoundTensor{graph->force_bind(*this), graph->force_bind(*this) + num_elements, shape_}.repr(out);
    if (!out) throw std::runtime_error(std::string("Error writing to file: ") + filename);
}

void Tensor::dump_now(const char* filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error(std::string("Cannot open file: ") + filename);
    size_t num_elements = 1;
    for (const auto &i: shape_) {
        num_elements *= i;
    }
    BoundTensor{graph->force_bind(*this), graph->force_bind(*this) + num_elements, shape_}.dump(out);
    if (!out) throw std::runtime_error(std::string("Error writing to file: ") + filename);
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

// ============= ПРОЧИЕ МЕТОДЫ =============

Tensor::Tensor(Graph *graph, const std::vector<size_t> &shape, const std::string &name) : graph(graph), id(++TENSOR_ID), shape_(shape), name_(name) {
    graph->names[id] = name;
}
Tensor &Tensor::operator[](const std::string_view &name) {
    if (!id) throw std::runtime_error("Empty tensor");
    name_ = name;
    graph->names[id] = name;
    return *this;
}

// ============= ПРЕОБРАЗОВАНИЕ В IndexList =============
IndexList Tensor::as_index(unsigned int axis) const {
    if (!id) throw std::runtime_error("Empty tensor");
    const CompatibleInt* ptr = graph->to_size_tensor(*this);
    // количество элементов = общее число элементов тензора
    if (shape_.size() > 1) throw std::runtime_error("It is impossible to make an array of indexes from a multidimensional array");
    return {ptr, shape_[0], axis};
}

// ============= ИНДЕКСЫ =============

Tensor Tensor::gather(const IndexList& indices) const {
    if (!id) throw std::runtime_error("Empty tensor");
    if (indices.data == nullptr) throw std::runtime_error("IndexList has null data");
    if (indices.size == 0) throw std::runtime_error("IndexList is empty");
    if (indices.axis >= shape_.size()) throw std::runtime_error("Gather axis out of bounds");

    std::vector<size_t> result_shape = shape_;
    result_shape[indices.axis] = indices.size;

    return graph->add_operation(OperationId::GATHER, *this, indices, result_shape);
}

// ============= РЕДУЦИРУЮЩИЕ ОПЕРАЦИИ =============

static std::vector<size_t> reduced_shape(const std::vector<size_t>& s, int axis, bool savedim = true) {
    int ndim = static_cast<int>(s.size());
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) throw std::runtime_error("Invalid reduction axis");
    std::vector<size_t> res = s;
    if (!savedim) {
        res.erase(res.begin() + axis);
    } else {
        res[axis] = 1;
    }
    return res;
}

Tensor Tensor::reduce_min(int axis, bool savedim) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::REDUCE_MIN, *this,
        static_cast<size_t>(axis < 0 ? axis + shape_.size() : axis),
        reduced_shape(shape_, axis, savedim)
    );
}

Tensor Tensor::reduce_max(int axis, bool savedim) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::REDUCE_MAX, *this,
        static_cast<size_t>(axis < 0 ? axis + shape_.size() : axis),
        reduced_shape(shape_, axis, savedim)
    );
}

Tensor Tensor::reduce_sum(int axis, bool savedim) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::REDUCE_SUM, *this,
        static_cast<size_t>(axis < 0 ? axis + shape_.size() : axis),
        reduced_shape(shape_, axis, savedim)
    );
}

Tensor Tensor::reduce_mean(int axis, bool savedim) const {
    if (!id) throw std::runtime_error("Empty tensor");
    return graph->add_operation(OperationId::REDUCE_MEAN, *this,
        static_cast<size_t>(axis < 0 ? axis + shape_.size() : axis),
        reduced_shape(shape_, axis, savedim)
    );
}

// ============= РАСШИРЕНИЕ ТЕНЗОРА =============
Tensor Tensor::repeat(size_t count) const {
    std::vector<size_t> res = shape_;
    res.back() = res.back() * count;
    return graph->add_operation(OperationId::REPEAT, *this, count, res);
}

// ============= СОСТАВНЫЕ ОПЕРАЦИИ =============
Tensor CDMath::softmax(const Tensor &t, int dim) {
    Tensor max_x = t.reduce_max(dim).repeat(t.shape().back());
    Tensor e = (t - max_x).exp();
    Tensor s = e.reduce_sum(dim).repeat(t.shape().back());
    return e / s;
}
