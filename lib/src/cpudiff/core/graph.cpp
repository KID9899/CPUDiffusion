//
// Created by iliya on 5/21/26.
//

#include <cstdlib>
#include <cstring>
#include <numeric>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "cpudiff/typeutils.h"
#include "cpudiff/safetensors.h"

#include "graph.h"

static size_t total_elements(const std::vector<uint64_t> &shape) {
    size_t ans = 1;
    for (const size_t &i: shape) {
        ans *= i;
    }
    return ans;
}

Graph::~Graph() {
    for (auto& p : maped) {
        free(p);
    }
}

Tensor Graph::add_existing(void *start, void *end, const std::vector<uint64_t> &shape, Dtype dtype) {
    if (dtype == FloatDtype) {
        tensors.push_back(std::make_unique<UniqueTensor>(this, shape));
        tensors.back()->set_memory(start, end);
        return tensors.back()->reference();
    } else {
        Tensor t = allocate(shape);
        dtype_to_float(static_cast<float*>(t.data()), start, total_elements(shape), dtype);
        return t;
    }
}

Tensor Graph::allocate(const std::vector<uint64_t> &shape) {
    Tensor t = future();
    alloc_promised(t.unique, shape);
    return t;
}

void Graph::add_operation(OperationId id, const UniqueTensor *src1, GraphOperation::SecondArg src2, UniqueTensor *result) {
    operations.emplace_back(id, src1, src2, result);
}

Tensor Graph::future() {
    tensors.push_back(std::make_unique<UniqueTensor>(this, std::vector<uint64_t>()));
    return tensors.back()->reference();
}

void Graph::alloc_promised(UniqueTensor *tensor, const std::vector<uint64_t> &shape) {
    if (tensor->data() != nullptr) throw std::runtime_error("It is impossible to allocate an existing tensor");
    // Обновляем размеры тензора
    tensor->shape_ = shape;

    // Выделяем память
    size_t num = total_elements(shape);
    size_t alloc_bytes = num * sizeof(float);

    void *new_data = malloc(alloc_bytes);
    if (!new_data) throw std::bad_alloc();

    // Заполняем нулями
    std::memset(new_data, 0, alloc_bytes);

    // Записываем выделенную память
    maped.emplace_back(new_data);
    void *new_end = static_cast<char*>(new_data) + alloc_bytes;

    // Устанавливаем тензор в реальный
    tensor->set_memory(new_data, new_end);
}

void Graph::dump_dot(const std::string &filename, const SafeTensorsFile *safetensors) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Строим мапу из доступных имён
    std::unordered_map<const void*, std::string> ptr_to_name;
    if (safetensors) {
        const auto &names = safetensors->names();
        const auto &tensors = safetensors->tensors();
        for (size_t i = 0; i < names.size(); ++i) {
            const Tensor &t = tensors[i];
            if (t.data() != nullptr) {
                ptr_to_name[t.data()] = names[i];
            }
        }
    }

    // Заголовок DOT-файла
    ofs << "digraph G {\n";
    ofs << "  rankdir=TB;\n";
    ofs << "  node [fontname=\"Courier New\"];\n\n";

    // Создаём узлы для тензоров
    for (const std::unique_ptr<UniqueTensor> &t : tensors) {
        if (t->data() == nullptr) continue;
        std::string id = "t_" + std::to_string(reinterpret_cast<uintptr_t>(t->data()));
        // Первая строка - имя или указатель, вторая - размерность
        std::string label;
        auto it = ptr_to_name.find(t->data());
        if (it != ptr_to_name.end()) {
            label = it->second;
        } else {
            std::stringstream ss;
            ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(t->data());
            label = ss.str();
        }
        // Формируем размерность
        std::string dims = "[";
        for (size_t i = 0; i < t->shape().size(); ++i) {
            if (i != 0) dims += ", ";
            dims += std::to_string(t->shape()[i]);
        }
        dims += "]";
        ofs << "  " << id << " [shape=box, label=\"" << label << "\\n" << dims << "\"];\n";
    }
    ofs << "\n";

    // Делаем ромбики для операций
    size_t op_idx = 0;
    for (const auto &op : operations) {
        std::string op_id = "op_" + std::to_string(op_idx++);
        ofs << "  " << op_id << " [shape=diamond, label=\"" << operation_name(op.id);
        // Если это константа, нужно добавить его в название операции
        if (op.id == OperationId::FMUL || op.id == OperationId::FDIV) ofs << op.src2.f;
        ofs << "\"];\n";

        // Ребро от src1 к операции
        if (op.src1) {
            std::string src_id = "t_" + std::to_string(reinterpret_cast<uintptr_t>(op.src1->data()));
            ofs << "  " << src_id << " -> " << op_id << ";\n";
        }

        // Ребро от src2 к операции, если это тензор
        if (op.id == OperationId::ADD  || op.id == OperationId::SUB ||
            op.id == OperationId::MUL  || op.id == OperationId::DIV ||
            op.id == OperationId::MATMUL) {
            if (op.src2.t) {
                std::string src2_id = "t_" + std::to_string(reinterpret_cast<uintptr_t>(op.src2.t->data()));
                ofs << "  " << src2_id << " -> " << op_id << ";\n";
            }
        }

        // Ребро от операции к результату
        if (op.result) {
            std::string res_id = "t_" + std::to_string(reinterpret_cast<uintptr_t>(op.result->data()));
            ofs << "  " << op_id << " -> " << res_id << ";\n";
        }
    }

    ofs << "}\n";
    ofs.close();
}
