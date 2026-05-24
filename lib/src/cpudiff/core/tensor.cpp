//
// Created by iliya on 5/22/26.
//

#include <optional>
#include <sstream>
#include <functional>
#include <iomanip>
#include <cmath>

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

Tensor UniqueTensor::reference() {
    return Tensor(this);
}

void UniqueTensor::dump(std::ostream &out) const {
    out.write(name_.c_str(), name_.size());
    out.put('\0');

    size_t rank = shape_.size();
    out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));

    out.write(reinterpret_cast<const char*>(shape_.data()), rank * sizeof(size_t));

    size_t total = 1;
    for (size_t d : shape_) total *= d;
    const float* data_ptr = static_cast<const float*>(start);
    out.write(reinterpret_cast<const char*>(data_ptr), total * sizeof(float) / sizeof(char));
}

void UniqueTensor::repr(std::ostream &out) const {
    out << "name: \"" << name_ << "\"\n";

    out << "shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) out << ", ";
        out << shape_[i];
    }
    out << "]\n\n";

    size_t total = 1;
    for (size_t d : shape_) total *= d;
    const float* data = static_cast<const float*>(start);

    size_t to_one_float = 1;
    for (size_t i = 0; i < total; ++i) {
        std::ostringstream oss;
        oss << data[i];
        to_one_float = std::max(to_one_float, oss.str().length());
    }

    std::function<void(size_t, size_t&, int)> print_dim = [&](size_t dim, size_t& index, int indent) {
        // Индекс предпоследнего измерения
        const size_t pre_last = shape_.size() >= 2 ? shape_.size() - 2 : 0;

        if (dim == shape_.size() - 1) {
            // Последнее измерение - печатаем числа в строку
            out << "[";
            for (size_t i = 0; i < shape_[dim]; ++i) {
                if (i > 0) out << ", ";
                out << std::setw(static_cast<int>(to_one_float)) << data[index++];
            }
            out << "]";
        } else if (shape_.size() >= 2 && dim == pre_last) {
            // Предпоследнее измерение: каждый элемент на новой строке
            out << "[";
            out << "\n";
            for (size_t i = 0; i < shape_[dim]; ++i) {
                out << std::string(indent + 4, ' ');
                print_dim(dim + 1, index, indent + 4);
                if (i != shape_[dim] - 1) {
                    out << ",";
                }
                out << "\n";
            }
            out << std::string(indent, ' ') << "]";
        } else {
            // Более высокие измерения
            out << "[";
            if (shape_[dim] > 0) {
                out << "\n" << std::string(indent + 4, ' ');
                for (size_t i = 0; i < shape_[dim]; ++i) {
                    if (i > 0) {
                        out << ", ";
                    }
                    print_dim(dim + 1, index, indent + 4);
                }
                out << "\n" << std::string(indent, ' ') << "]";
            } else {
                out << "]";
            }
        }
    };

    size_t idx = 0;
    print_dim(0, idx, 0);
}

Tensor &Tensor::operator<<(const std::vector<size_t> &shape) {
    if (unique != nullptr && unique->start == nullptr) { graph()->alloc_promised(unique, shape); }
    if (this->shape() != shape) throw std::runtime_error("It is not possible to set a size other than the current one");
    return *this;
}

Tensor &Tensor::operator=(Tensor::CanAssign other) {
    if (std::holds_alternative<TensorResult>(other)) {
        TensorResult &res = std::get<TensorResult>(other);
        if (graph() != res.src1->graph)
            throw std::runtime_error("An attempt to link tensors from different computational graphs");
        // В случае, если тензор обещанный
        if (unique != nullptr && unique->start == nullptr) { graph()->alloc_promised(unique, res.shape); }
        if (res.shape != shape())
            throw std::runtime_error(
                    "The dimensions of the recorded tensor " + shape_to_string(res.shape) + " and container " +
                    shape_to_string(shape()) + " are different");
        if (res.id == OperationId::MATMUL && (res.src1 == unique || res.src2.t == unique))
            throw std::runtime_error(
                    "You cannot perform non-piecemeal operations with writing the result to one of the arguments");
        if (res.id == OperationId::COPY && res.src1 == unique)
            throw std::runtime_error("It is forbidden to copy a tensor into itself");
        graph()->add_operation(res.id, res.src1, res.src2, unique);
    } else if (std::holds_alternative<TensorView>(other)) {
        TensorView &view = std::get<TensorView>(other);
        if (graph() != view.unique->graph)
            throw std::runtime_error("An attempt to link tensors from different computational graphs");
        // В случае, если тензор обещанный
        if (unique->start == nullptr) {
            unique->shape_ = view.new_shape;
            unique->set_memory(view.unique->start, view.unique->end);
        }
        if (unique->start != view.unique->start)
            throw std::runtime_error("It is not possible to assign a TensorView to a tensor bound to a memory location other than the one passed to TensorView");
        else if (unique->shape() != view.new_shape)
            throw std::runtime_error("It is not possible to assign a TensorView to a tensor that implements a different representation of this data (a different shape)");
        graph()->add_operation(OperationId::VIEW, view.unique, {.t = nullptr}, unique);
    }
    return *this;
}


void *Tensor::data() const {
    if (!unique) throw std::runtime_error("An attempt to use an unloaded tensor");
    return unique->data();
}
const std::vector<size_t> &Tensor::shape() const {
    if (!unique) throw std::runtime_error("An attempt to use an unloaded tensor");
    return unique->shape();
}
Graph *Tensor::graph() {
    if (!unique) throw std::runtime_error("An attempt to use an unloaded tensor");
    return unique->graph;
}

void Tensor::dump(const char *filename) {
    if (!unique) throw std::runtime_error("An attempt to use an unloaded tensor");
    graph()->add_operation(OperationId::DUMP, unique, {.s = filename}, nullptr);
}
void Tensor::repr(const char *filename) {
    if (!unique) throw std::runtime_error("An attempt to use an unloaded tensor");
    graph()->add_operation(OperationId::REPR, unique, {.s = filename}, nullptr);
}

Tensor::Tensor(UniqueTensor *unique) : unique(unique) {}

// ============= ВСЯКИЕ ОПЕРАЦИИ =============

void Tensor::fill(float v) { this->operator=(TensorResult{OperationId::FILL, unique, {.f = v}, shape()}); }
void Tensor::randn() { this->operator=(TensorResult{OperationId::RANDN, unique, {.t = nullptr}, shape()}); }
TensorResult Tensor::copy() const { return {OperationId::COPY, unique, {.t = nullptr}, shape()}; }
TensorResult Tensor::transpose() const {
    auto res_shape = transpose_shape(shape());
    if (!res_shape) throw std::runtime_error("It is impossible to transpose a tensor: " + shape_to_string(shape()));
    return {OperationId::TRANSPOSE, unique, {.t = nullptr}, *res_shape};
}


TensorResult Tensor::operator+(Tensor other) const {
    auto res_shape = can_broadcast_shapes(shape(), other.shape());
    if (!res_shape) throw std::runtime_error("Incompatible shapes for element-wise addition: " + shape_to_string(shape()) + " and "  + shape_to_string(other.shape()));
    return {OperationId::ADD, unique, {.t = other.unique}, shape()};
}
TensorResult Tensor::operator-(Tensor other) const {
    auto res_shape = can_broadcast_shapes(shape(), other.shape());
    if (!res_shape) throw std::runtime_error("Incompatible shapes for element-wise subtraction: " + shape_to_string(shape()) + " and "  + shape_to_string(other.shape()));
    return {OperationId::SUB, unique, {.t = other.unique}, shape()};
}
TensorResult Tensor::operator-() const { return {OperationId::NEGATE, unique, {.t = nullptr}, shape()}; }

TensorResult Tensor::operator*(Tensor other) const {
    auto res_shape = can_broadcast_shapes(shape(), other.shape());
    if (!res_shape) throw std::runtime_error("Incompatible shapes for element-wise multiplication: " + shape_to_string(shape()) + " and "  + shape_to_string(other.shape()));
    return {OperationId::MUL, unique, {.t = other.unique}, shape()};
}
TensorResult Tensor::operator*(float other) const { return {OperationId::FMUL, unique, {.f = other}, shape()}; }
TensorResult Tensor::operator/(Tensor other) const {
    auto res_shape = can_broadcast_shapes(shape(), other.shape());
    if (!res_shape) throw std::runtime_error("Incompatible shapes for element-wise division: " + shape_to_string(shape()) + " and "  + shape_to_string(other.shape()));
    return {OperationId::DIV, unique, {.t = other.unique}, shape()};
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

TensorView Tensor::view(const std::vector<size_t> &shape) const {
    // общее количество элементов исходного тензора
    size_t expected = 1;
    for (size_t d : this->shape()) expected *= d;

    size_t known_product = 1;   // произведение известных размерностей
    size_t infer_index = 0;     // позиция, где стоит 0
    size_t infer_count = 0;     // сколько раз встретился 0

    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) {
            infer_index = i;
            ++infer_count;
        } else {
            known_product *= shape[i];
        }
    }

    if (infer_count == 0) {
        // без авто-размерности
        size_t total = 1;
        for (size_t d : shape) total *= d;
        if (expected != total) {
            throw std::runtime_error(
                    "Reinterpretation of the tensor is impossible because the transmitted "
                    "and expected number of elements do not match: " +
                    std::to_string(total) + " vs " + std::to_string(expected));
        }
        return {unique, shape};
    }
    else if (infer_count == 1) {
        // автоматически подсчитать одну размерность
        if (known_product == 0 || expected % known_product != 0) {
            throw std::runtime_error(
                    "Cannot infer dimension: total elements " + std::to_string(expected) +
                    " is not divisible by product of known dimensions " +
                    std::to_string(known_product));
        }
        size_t inferred = expected / known_product;
        std::vector<size_t> full_shape = shape;
        full_shape[infer_index] = inferred;
        return {unique, full_shape};
    }
    else {
        throw std::runtime_error("Only one dimension can be inferred (multiple -1 found)");
    }
}
TensorView Tensor::flatten() const {
    size_t second = 1;
    const std::vector<size_t> &s = shape();
    for (size_t i = 1; i < s.size(); ++i) second *= s[i];
    return view({0, second});
}


// ============= TensorResult =============

TensorResult::TensorResult(OperationId id, const UniqueTensor *src1, GraphOperation::SecondArg src2, const std::vector<size_t> &shape)
    : id(id), src1(src1), src2(src2), shape(shape) {}
