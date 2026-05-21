//
// Created by iliya on 5/22/26.
//

#include <optional>
#include <sstream>

#include "graph.h"
#include "tensor.h"

// Проверка на возможность поэлеметных операций с данными тензорами
static std::optional<std::vector<uint64_t>> broadcast_shapes(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    size_t rank = std::max(a.size(), b.size());
    std::vector<uint64_t> result(rank);

    for (size_t i = 0; i < rank; ++i) {
        uint64_t dim_a = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        uint64_t dim_b = (i < b.size()) ? b[b.size() - 1 - i] : 1;
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return std::nullopt;
        }
        result[rank - 1 - i] = std::max(dim_a, dim_b);
    }
    return result;
}

// Вычисление размерности матричного умножения (если оно возможно)
static std::optional<std::vector<uint64_t>> matmul_shape(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    // Матричное умножение требует хотя бы одной размерности у каждого аргумента
    if (a.empty() || b.empty()) return std::nullopt;

    // Оба одномерные — скалярное произведение
    if (a.size() == 1 && b.size() == 1) {
        if (a[0] != b[0]) return std::nullopt;
        return std::vector<uint64_t>{};  // скаляр (0-мерный тензор)
    }

    // Один одномерный, второй двумерный
    if (a.size() == 1 && b.size() == 2) {
        if (a[0] != b[0]) return std::nullopt;
        return std::vector<uint64_t>{b[1]};  // (n) @ (n,p) -> (p)
    }
    if (a.size() == 2 && b.size() == 1) {
        if (a[1] != b[0]) return std::nullopt;
        return std::vector<uint64_t>{a[0]};  // (m,n) @ (n) -> (m)
    }

    // Оба двумерные — классическое умножение матриц
    if (a.size() == 2 && b.size() == 2) {
        if (a[1] != b[0]) return std::nullopt;
        return std::vector<uint64_t>{a[0], b[1]};  // (m,n) @ (n,p) -> (m,p)
    }

    // Пакетное матричное умножение (хотя бы один >2D, либо 1D с >2D)
    // Превращаем одномерные аргументы в двумерные матрицы
    bool a_was_1d = (a.size() == 1);
    bool b_was_1d = (b.size() == 1);

    std::vector<uint64_t> a_mat = a_was_1d ? std::vector<uint64_t>{1, a[0]} : a;
    std::vector<uint64_t> b_mat = b_was_1d ? std::vector<uint64_t>{b[0], 1} : b;

    // Разделяем на пакетные размерности и матричные
    // a_mat = batch_a + [M, K]
    // b_mat = batch_b + [K, N]
    if (a_mat.size() < 2 || b_mat.size() < 2) return std::nullopt;

    std::vector<uint64_t> batch_a(a_mat.begin(), a_mat.end() - 2);
    std::vector<uint64_t> batch_b(b_mat.begin(), b_mat.end() - 2);

    uint64_t M = a_mat[a_mat.size() - 2];
    uint64_t K_a = a_mat[a_mat.size() - 1];
    uint64_t K_b = b_mat[b_mat.size() - 2];
    uint64_t N = b_mat[b_mat.size() - 1];

    if (K_a != K_b) return std::nullopt;  // внутренние размерности матриц не совпадают

    // Пакетные размерности должны быть совместимы по broadcast
    auto batch_res = broadcast_shapes(batch_a, batch_b);
    if (!batch_res) return std::nullopt;

    // Собираем полную форму результата: батч + [M, N]
    std::vector<uint64_t> result = std::move(*batch_res);
    result.push_back(M);
    result.push_back(N);


    if (a_was_1d) {
        result.erase(result.end() - 2);
    }
    if (b_was_1d) {
        result.pop_back();
    }
    return result;
}

static std::string shape_to_string(const std::vector<uint64_t>& s) {
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
static std::optional<std::vector<uint64_t>> transpose_shape(
        const std::vector<uint64_t>& shape,
        int64_t dim0 = -2,
        int64_t dim1 = -1
) {
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;

    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        return std::nullopt;
    }

    std::vector<uint64_t> new_shape = shape;
    std::swap(new_shape[dim0], new_shape[dim1]);

    return new_shape;
}

// ============= БАЗОВЫЕ ФУНКЦИИ ТЕНЗОРОВ =============

Tensor UniqueTensor::reference() {
    return Tensor(this);
}

Tensor &Tensor::operator=(TensorResult res) {
    if (graph() != res.src1->graph) throw std::runtime_error("An attempt to link tensors from different computational graphs");
    // В случае, если тензор обещанный
    if (unique != nullptr && unique->start == nullptr) { graph()->alloc_promised(unique, res.shape); }
    if (res.shape != shape()) throw std::runtime_error("The dimensions of the recorded tensor " + shape_to_string(res.shape) + " and container " + shape_to_string(shape()) + " are different");
    if (res.id == OperationId::MATMUL && (res.src1 == unique || res.src2.t == unique)) throw std::runtime_error("You cannot perform non-piecemeal operations with writing the result to one of the arguments");
    if (res.id == OperationId::COPY && res.src1 == unique) throw std::runtime_error("It is forbidden to copy a tensor into itself");
    graph()->add_operation(res.id, res.src1, res.src2, unique);
    return *this;
}


void *Tensor::data() const {
    if (!unique) throw std::runtime_error("An attempt to use an unloaded tensor");
    return unique->data();
}
const std::vector<uint64_t> &Tensor::shape() const {
    if (!unique) throw std::runtime_error("An attempt to use an unloaded tensor");
    return unique->shape();
}
Graph *Tensor::graph() {
    if (!unique) throw std::runtime_error("An attempt to use an unloaded tensor");
    return unique->graph;
}

Tensor::Tensor(UniqueTensor *unique) : unique(unique) {}

// ============= ВСЯКИЕ ОПЕРАЦИИ =============

void Tensor::fill(float v) { this->operator=({OperationId::FILL, unique, {.f = v}, shape()}); }
void Tensor::randn() { this->operator=({OperationId::RANDN, unique, {.t = nullptr}, shape()}); }
TensorResult Tensor::copy() const { return {OperationId::COPY, unique, {.t = nullptr}, shape()}; }
TensorResult Tensor::transpose() const {
    auto res_shape = transpose_shape(shape());
    if (!res_shape) throw std::runtime_error("It is impossible to transpose a tensor: " + shape_to_string(shape()));
    return {OperationId::TRANSPOSE, unique, {.t = nullptr}, *res_shape};
}


TensorResult Tensor::operator+(Tensor other) const {
    auto res_shape = broadcast_shapes(shape(), other.shape());
    if (!res_shape) throw std::runtime_error("Incompatible shapes for element-wise addition: " + shape_to_string(shape()) + " and "  + shape_to_string(other.shape()));
    return {OperationId::ADD, unique, {.t = other.unique}, *res_shape};
}
TensorResult Tensor::operator-(Tensor other) const {
    auto res_shape = broadcast_shapes(shape(), other.shape());
    if (!res_shape) throw std::runtime_error("Incompatible shapes for element-wise subtraction: " + shape_to_string(shape()) + " and "  + shape_to_string(other.shape()));
    return {OperationId::SUB, unique, {.t = other.unique}, *res_shape};
}
TensorResult Tensor::operator-() const { return {OperationId::NEGATE, unique, {.t = nullptr}, shape()}; }

TensorResult Tensor::operator*(Tensor other) const {
    auto res_shape = broadcast_shapes(shape(), other.shape());
    if (!res_shape) throw std::runtime_error("Incompatible shapes for element-wise multiplication: " + shape_to_string(shape()) + " and "  + shape_to_string(other.shape()));
    return {OperationId::MUL, unique, {.t = other.unique}, *res_shape};
}
TensorResult Tensor::operator*(float other) const { return {OperationId::FMUL, unique, {.f = other}, shape()}; }
TensorResult Tensor::operator/(Tensor other) const {
    auto res_shape = broadcast_shapes(shape(), other.shape());
    if (!res_shape) throw std::runtime_error("Incompatible shapes for element-wise division: " + shape_to_string(shape()) + " and "  + shape_to_string(other.shape()));
    return {OperationId::DIV, unique, {.t = other.unique}, *res_shape};
}
TensorResult Tensor::operator/(float other) const { return {OperationId::FDIV, unique, {.f = other}, shape()}; }
TensorResult Tensor::invert() const { return {OperationId::INVERT, unique, {.t = nullptr}, shape()}; }

TensorResult Tensor::operator^(Tensor other) const {
    auto res_shape = matmul_shape(shape(), other.shape());
    if (!res_shape) throw std::runtime_error("Incompatible shapes for matrix multiplication: " + shape_to_string(shape()) + " and "  + shape_to_string(other.shape()));
    return {OperationId::MATMUL, unique, {.t = other.unique}, *res_shape};
}

TensorResult Tensor::exp() const { return {OperationId::EXP, unique, {.t = nullptr}, shape()}; }
TensorResult Tensor::relu() const { return {OperationId::RELU, unique, {.t = nullptr}, shape()}; }
TensorResult Tensor::sigmoid() const { return {OperationId::SIGMOID, unique, {.t = nullptr}, shape()}; }

// ============= TensorResult =============

TensorResult::TensorResult(OperationId id, const UniqueTensor *src1, GraphOperation::SecondArg src2, const std::vector<uint64_t> &shape)
    : id(id), src1(src1), src2(src2), shape(shape) {}
Tensor &TensorResult::operator[](Tensor &t) { return t = *this; }
