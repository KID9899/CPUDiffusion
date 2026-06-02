//
// Created by iliya on 5/21/26.
//

#include <functional>
#include <fstream>
#include <ostream>
#include <sstream>
#include <iomanip>

#include "cpudiff/typeutils.h"
#include "cpudiff/safetensors.h"

#include "graph.h"

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

static size_t num_elements(const std::vector<size_t> &shape) {
    if (shape.empty()) return 0;
    size_t ans = 1;
    for (const size_t &i: shape) {
        ans *= i;
    }
    return ans;
}

Tensor Graph::allocate(const std::vector<size_t> &shape, const std::string &name) {
    if (compiled)
        throw std::runtime_error("Cannot allocate in a compiled graph");
    if (shape.empty())
        throw std::runtime_error("Empty shape");
    for (size_t d : shape)
        if (d == 0)
            throw std::runtime_error("Zero dimension in shape");

    size_t nelem = num_elements(shape);
    auto* ptr = new float[nelem]; // выделяем
    mapped.push_back(ptr); // владеем
    usable[ptr] = nelem; // доступный блок

    Tensor t(this, shape, name);
    bound[t.id] = ptr;
    return t;
}

Tensor Graph::link(float *data, const std::vector<size_t> &shape, const std::string &name) {
    if (compiled)
        throw std::runtime_error("Cannot link in a compiled graph");
    if (!data)
        throw std::runtime_error("Null data pointer");

    size_t nelem = num_elements(shape);
    usable[data] = nelem;               // доступный блок

    Tensor t(this, shape, name);
    bound[t.id] = data;
    return t;
}

float* Graph::force_bind(const Tensor &t, bool result) {
    if (t.id == 0 || t.graph != this)
        throw std::runtime_error("Cannot bind an empty or foreign tensor");

    if (compiled && !results.contains(t.id)) {
        throw std::runtime_error("Only the resulting tensors can be bound from a compiled graph");
    } else if (result) {
        if (size_tensors.contains(t.id)) throw std::runtime_error("The index tensor cannot be the resulting one");
        results.insert(t.id);
    }

    auto it = bound.find(t.id);
    if (it != bound.end())
        return it->second;              // память уже есть

    // Памяти ещё нет – выделяем сейчас
    size_t nelem = num_elements(t.shape_);
    auto* ptr = new float[nelem];
    mapped.push_back(ptr);               // владеем
    usable[ptr] = nelem;                // доступный блок
    bound[t.id] = ptr;
    return ptr;
}

Graph::~Graph() {
    if (current_graph == this) current_graph = nullptr;
    for (void* ptr : mapped)
        ::operator delete(ptr);
}

void Graph::compile() {
    if (compiled)
        throw std::runtime_error("Graph already compiled");
    if (promised.empty())
        throw std::runtime_error("No operations to compile");

    const size_t N = promised.size();

    // Вычисляем время жизни каждого id
    struct LifeInfo {
        int last_use    = -1;
        int creation_op = -1;
    };
    std::unordered_map<unsigned int, LifeInfo> life;

    for (const auto& [id, ptr] : bound)
        life[id];

    for (size_t i = 0; i < N; ++i) {
        const IncompleteOperation& op = promised[i];

        auto update = [&](unsigned int arg_id) {
            if (arg_id == 0) return;
            LifeInfo& info = life[arg_id];
            info.last_use = static_cast<int>(i);
            if (size_tensors.contains(arg_id)) throw std::runtime_error("It is impossible to perform tensor operations with the size tensor");
        };

        update(op.arg1);
        if (OperationGroups::arg2_is_tensor.contains(op.id))
            update(std::get<unsigned int>(op.arg2));

        if (!OperationGroups::no_result.contains(op.id) && op.result != 0)
            life[op.result].creation_op = static_cast<int>(i);
    }

    // Храним пары (указатель, размер в float)
    std::vector<std::pair<float*, size_t>> free_blocks;
    std::unordered_set<float*> used_ptrs;
    for (const auto& [id, ptr] : bound) used_ptrs.insert(ptr);
    for (const auto& [ptr, sz] : usable) {
        if (!used_ptrs.count(ptr))
            free_blocks.emplace_back(ptr, sz);
    }

    // Ищем наименьший подходящий свободный блок
    auto best_fit_free = [&](size_t required) -> float* {
        float* best_ptr = nullptr;
        size_t best_size = SIZE_MAX;
        size_t best_idx = 0;
        for (size_t i = 0; i < free_blocks.size(); ++i) {
            auto [ptr, sz] = free_blocks[i];
            if (sz >= required && sz < best_size) {
                best_ptr = ptr;
                best_size = sz;
                best_idx = i;
            }
        }
        if (best_ptr) {
            // Блок забирается целиком
            free_blocks[best_idx] = free_blocks.back();
            free_blocks.pop_back();
        }
        return best_ptr;
    };

    // Основной проход по операциям
    for (size_t i = 0; i < N; ++i) {
        const IncompleteOperation& op = promised[i];

        // Аргументы
        BoundTensor arg1_bt;
        if (op.arg1 != 0) {
            auto it = bound.find(op.arg1);
            if (it == bound.end())
                throw std::runtime_error("Op " + std::to_string(i) + ": arg1 missing memory");
            float* ptr = it->second;
            arg1_bt = {ptr, ptr + usable.at(ptr), op.arg1_shape};
        }

        CompleteOperation::ARG2_t arg2_val;
        if (OperationGroups::arg2_is_tensor.contains(op.id)) {
            unsigned int a2 = std::get<unsigned int>(op.arg2);
            if (size_tensors.contains(a2))
                throw std::runtime_error("Size tensor used in incompatible operation");
            if (a2 == 0) throw std::runtime_error("Null tensor arg2");
            auto it = bound.find(a2);
            if (it == bound.end())
                throw std::runtime_error("Op " + std::to_string(i) + ": arg2 missing memory");
            float* ptr = it->second;
            arg2_val = BoundTensor{ptr, ptr + usable.at(ptr), op.arg2_shape};
        } else if (OperationGroups::arg2_is_float.contains(op.id)) {
            arg2_val = std::get<float>(op.arg2);
        } else if (OperationGroups::arg2_is_string.contains(op.id)) {
            arg2_val = std::get<const char*>(op.arg2);
        } else if (OperationGroups::arg2_is_indexlist.contains(op.id)) {
            arg2_val = std::get<IndexList>(op.arg2);
        } else if (OperationGroups::arg2_is_size.contains(op.id)) {
            arg2_val = static_cast<size_t>(std::get<unsigned int>(op.arg2));
        } else {
            arg2_val = size_t{0u};
        }

        // Результат
        BoundTensor result_bt;
        if (OperationGroups::no_result.contains(op.id)) {
            // операции без результата
            result_bt = arg1_bt;
        } else {
            unsigned int res_id = op.result;
            size_t required = num_elements(op.result_shape);
            float* res_ptr = nullptr;

            if (OperationGroups::must_reuse_arg1.contains(op.id)) {
                // Результат обязан использовать ту же память, что и arg1
                auto it = bound.find(op.arg1);
                if (it == bound.end())
                    throw std::runtime_error("Op " + std::to_string(i) + ": arg1 has no memory for but it must reuse arg1");
                res_ptr = it->second;
                // Привязываем результат к той же памяти
                bound[op.result] = res_ptr;
            } else {
                auto bound_it = bound.find(res_id);
                if (bound_it != bound.end()) {
                    // Память уже выделена
                    res_ptr = bound_it->second;
                    if (usable.at(res_ptr) < required)
                        throw std::runtime_error("Pre-bound result too small");
                } else {
                    // 1) Переиспользование аргумента, если разрешено
                    if (OperationGroups::may_reuse_args.contains(op.id)) {
                        auto try_reuse = [&](unsigned int cand_id) -> bool {
                            if (cand_id == 0) return false;
                            LifeInfo &l = life[cand_id];
                            if (l.last_use != static_cast<int>(i)) return false;
                            if (results.count(cand_id)) return false;
                            float *cand_ptr = bound.at(cand_id);
                            size_t cand_size = usable.at(cand_ptr);
                            if (cand_size != required) return false;  // Только точное совпадение
                            bound[res_id] = cand_ptr; // Забираем блок
                            bound.erase(cand_id);
                            res_ptr = cand_ptr;
                            return true;
                        };
                        // Пробуем arg1, потом arg2
                        if (!try_reuse(op.arg1) && OperationGroups::arg2_is_tensor.contains(op.id))
                            try_reuse(std::get<unsigned int>(op.arg2));
                    }

                    // 2) Поиск среди свободных блоков
                    if (!res_ptr) {
                        res_ptr = best_fit_free(required);
                        if (res_ptr) {
                            bound[res_id] = res_ptr;
                        }
                    }

                    // 3) Новая память
                    if (!res_ptr) {
                        res_ptr = new float[required];
                        mapped.push_back(res_ptr);
                        usable[res_ptr] = required;
                        bound[res_id] = res_ptr;
                    }
                }
            }
            result_bt = {res_ptr, res_ptr + required, op.result_shape};
        }

        // Готовая операция
        CompleteOperation comp_op{op.id, arg1_bt, arg2_val, result_bt};
        operations.push_back(comp_op);

        // Освобождение аргументов, которые больше не нужны
        auto release = [&](unsigned int arg_id) {
            if (arg_id == 0) return;
            // Не освобождаем, если это финальный тензор или использован в будущем
            if (life[arg_id].last_use != static_cast<int>(i)) return;
            if (results.count(arg_id)) return;
            auto it = bound.find(arg_id);
            if (it == bound.end()) return;
            float* ptr = it->second;
            size_t sz = usable.at(ptr);
            free_blocks.emplace_back(ptr, sz);
        };

        release(op.arg1);
        if (OperationGroups::arg2_is_tensor.contains(op.id))
            release(std::get<unsigned int>(op.arg2));
    }

    compiled = true;
}

void Graph::repr_compiled(std::ostream& os) const {
    if (!compiled)
        throw std::runtime_error("Graph must be compiled before repr_compiled");

    // Собираем все указатели, участвующие в операциях
    std::vector<const float*> ptr_order;
    std::unordered_map<const float*, int> ptr_to_num;
    auto get_num = [&](const float* p) -> int {
        if (!p) return 0;
        auto it = ptr_to_num.find(p);
        if (it == ptr_to_num.end()) {
            int num = static_cast<int>(ptr_order.size()) + 1;
            ptr_order.push_back(p);
            ptr_to_num[p] = num;
            return num;
        }
        return it->second;
    };

    for (const auto& op : operations) {
        get_num(op.arg1.start);
        if (auto* bt = std::get_if<BoundTensor>(&op.arg2)) {
            get_num(bt->start);
        }
        get_num(op.result.start);
    }

    // Была ли запись в этот участок памяти
    std::unordered_set<const float*> defined;

    // Указатель ещё не определён
    auto ensure_defined = [&](const float* p) {
        if (!p) return;
        if (defined.find(p) == defined.end()) {
            os << "use [" << get_num(p) << "]\n";
            defined.insert(p);
        }
    };

    // Выводим операции
    for (const auto& op : operations) {
        const float* arg1_ptr = op.arg1.start;
        ensure_defined(arg1_ptr);

        const float* arg2_ptr = nullptr;
        if (auto* bt = std::get_if<BoundTensor>(&op.arg2)) {
            arg2_ptr = bt->start;
            ensure_defined(arg2_ptr);
        }

        const float* res_ptr = op.result.start;
        const bool has_result = !OperationGroups::no_result.contains(op.id);

        // Вывод левой части, если есть
        if (has_result) {
            os << "[" << get_num(res_ptr) << "] = ";
        }

        // Вывод операции
        if (OperationGroups::arg2_is_float.contains(op.id)) {
            float val = std::get<float>(op.arg2);
            os << "[" << get_num(arg1_ptr) << "] " << operation_name(op.id) << " " << val << "\n";
        }
        else if (OperationGroups::arg2_is_string.contains(op.id)) {
            const char* str = std::get<const char*>(op.arg2);
            os << operation_name(op.id) << " [" << get_num(arg1_ptr) << "] to \"" << str << "\"\n";
        }
        else if (OperationGroups::arg2_is_tensor.contains(op.id)) {
            os << "[" << get_num(arg1_ptr) << "] " << operation_name(op.id) << " [" << get_num(arg2_ptr) << "]\n";
        }
        else if (OperationGroups::arg2_is_null.contains(op.id)) {
            os << operation_name(op.id) << " [" << get_num(arg1_ptr) << "]\n";
        }
        else if (OperationGroups::arg2_is_indexlist.contains(op.id)) {
            const auto& idx = std::get<IndexList>(op.arg2);
            os << "[" << get_num(arg1_ptr) << "] gather axis=" << idx.axis << " len=" << idx.size << "\n";
        }
        else if (OperationGroups::arg2_is_size.contains(op.id)) {
            size_t axis = std::get<size_t>(op.arg2);
            os << operation_name(op.id) << " [" << get_num(arg1_ptr) << "] at " << axis << "\n";
        }
        else {
            os << "? unknown operation\n";
        }

        // После операции результат становится определённым
        if (has_result) {
            defined.insert(res_ptr);
        }
    }

    // Вывод result для финальных тензоров
    for (unsigned int res_id : results) {
        auto it = bound.find(res_id);
        if (it != bound.end()) {
            const float* ptr = it->second;
            if (ptr && defined.count(ptr)) {
                os << "result [" << get_num(ptr) << "]\n";
            }
        }
    }
}

void Graph::repr_compiled(const char* filename) const {
    std::ofstream out(filename);
    if (!out)
        throw std::runtime_error(std::string("Cannot open file: ") + filename);
    repr_compiled(out);
}


void Graph::repr(const std::string &filename, const std::string &fontname) const {
    std::ofstream ofs(filename);
    if (!ofs) throw std::runtime_error("Cannot open file: " + filename);

    ofs << "digraph G {\n";
    ofs << "  rankdir=TB;\n";
    ofs << "  node [fontname=\"" << fontname << "\"];\n\n";

    // Информация о тензорах
    struct TensorInfo {
        std::vector<size_t> shape;
        bool is_result = false;
        bool is_initial = false;
    };
    std::unordered_map<unsigned int, TensorInfo> tinfo;

    // Регистрируем все id из promised
    for (const auto& op : promised) {
        if (op.arg1) tinfo[op.arg1];
        if (OperationGroups::arg2_is_tensor.contains(op.id))
            tinfo[std::get<unsigned int>(op.arg2)];
        if (!OperationGroups::no_result.contains(op.id) && op.result)
            tinfo[op.result];
    }

    // Операции
    for (const auto& op : promised) {
        if (op.arg1) tinfo[op.arg1].shape = op.arg1_shape;
        if (OperationGroups::arg2_is_tensor.contains(op.id)) {
            unsigned int a2 = std::get<unsigned int>(op.arg2);
            if (a2) tinfo[a2].shape = op.arg2_shape;
        }
        if (!OperationGroups::no_result.contains(op.id) && op.result)
            tinfo[op.result].shape = op.result_shape;
    }

    // Начальные: используются до первой записи
    std::unordered_set<unsigned int> written;
    for (const auto& op : promised) {
        auto check = [&](unsigned int id) {
            if (id && written.find(id) == written.end())
                tinfo[id].is_initial = true;
        };
        check(op.arg1);
        if (OperationGroups::arg2_is_tensor.contains(op.id))
            check(std::get<unsigned int>(op.arg2));

        if (!OperationGroups::no_result.contains(op.id) && op.result)
            written.insert(op.result);
        else if (op.id == OperationId::FILL || op.id == OperationId::RANDN)
            written.insert(op.arg1);
    }

    // Конечные - только те, что в results
    for (unsigned int rid : results)
        if (tinfo.count(rid)) tinfo[rid].is_result = true;

    // Узлы тензоров с заливкой
    for (const auto& [id, info] : tinfo) {
        std::stringstream label;
        auto it = names.find(id);
        if (it == names.end()) {
            label << "id:" << id;
        } else {
            label << it->second;
        }
        label << "\\n" << shape_to_string(info.shape);
        std::string fillcolor = "white";
        if (info.is_initial) {
            fillcolor = "pink";
        } else if (info.is_result) {
            fillcolor = "lightgreen";
        }
        ofs << "  tensor_" << id
            << " [shape=box, style=filled, fillcolor=" << fillcolor
            << ", label=\"" << label.str() << "\"];\n";
    }

    // Операции и рёбра
    for (size_t idx = 0; idx < promised.size(); ++idx) {
        const auto& op = promised[idx];
        std::string op_id = "op_" + std::to_string(idx);
        std::string op_label = operation_name(op.id);

        if (OperationGroups::arg2_is_float.contains(op.id)) {
            float val = std::get<float>(op.arg2);
            std::stringstream ss;
            ss << " " << val;
            op_label += ss.str();
        } else if (OperationGroups::arg2_is_string.contains(op.id)) {
            op_label += " \\\"" + std::string(std::get<const char*>(op.arg2)) + "\\\"";
        } else if (OperationGroups::arg2_is_size.contains(op.id)) {
            unsigned int axis = std::get<unsigned int>(op.arg2);
            op_label += " at " + std::to_string(axis);
        }

        ofs << "  " << op_id << " [shape=diamond, label=\"" << op_label << "\"];\n";

        if (op.arg1)
            ofs << "  tensor_" << op.arg1 << " -> " << op_id << ";\n";
        if (OperationGroups::arg2_is_tensor.contains(op.id)) {
            unsigned int a2 = std::get<unsigned int>(op.arg2);
            if (a2)
                ofs << "  tensor_" << a2 << " -> " << op_id << ";\n";
        }
        if (OperationGroups::arg2_is_indexlist.contains(op.id)) {
            const auto& idl = std::get<IndexList>(op.arg2);
            if (!compiled || !usable.contains(const_cast<float*>(reinterpret_cast<const float*>(idl.data)))) {
                ofs << "block_" << std::to_string(reinterpret_cast<uintptr_t>(idl.data)) << " [shape=box, style=filled, fillcolor=lightblue, label=\"indexes\"];\n";
            }
            ofs << "block_" << std::to_string(reinterpret_cast<uintptr_t>(idl.data)) << " -> " << op_id << ";\n";
        }
        if (!OperationGroups::no_result.contains(op.id) && op.result)
            ofs << "  " << op_id << " -> tensor_" << op.result << ";\n";
    }

    // Блоки памяти и привязки
    if (compiled) {
        // Собираем уникальные блоки из bound
        std::unordered_map<float*, size_t> blocks;
        for (const auto& [id, ptr] : bound)
            blocks[ptr] = usable.at(ptr);

        for (const auto& [ptr, sz] : blocks) {
            std::string block_id =
                    "block_" + std::to_string(reinterpret_cast<uintptr_t>(ptr));
            std::stringstream label;
            label << "0x" << std::hex << reinterpret_cast<uintptr_t>(ptr)
                  << "\\n" << std::dec << sz << " items";
            ofs << "  " << block_id
                << " [shape=box, style=filled, fillcolor=lightblue, label=\""
                << label.str() << "\"];\n";

            // Пунктирные стрелки от тензоров к блоку
            for (const auto& [tid, t_ptr] : bound) {
                if (t_ptr == ptr && tinfo.count(tid))
                    ofs << "  tensor_" << tid << " -> " << block_id
                        << " [style=dashed, color=blue];\n";
            }
        }
    }

    ofs << "}\n";
}

void BoundTensor::repr(std::ostream &out) const {
    out << "shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) out << ", ";
        out << shape[i];
    }
    out << "]\n\n";

    size_t total = num_elements(shape);
    const float* data = start;

    // Определяем ширину поля для выравнивания
    size_t max_width = 1;
    for (size_t i = 0; i < total; ++i) {
        std::ostringstream oss;
        oss << data[i];
        max_width = std::max(max_width, oss.str().length());
    }

    std::function<void(size_t, size_t&, int)> print_dim = [&](size_t dim, size_t& index, int indent) {
        const size_t pre_last = shape.size() >= 2 ? shape.size() - 2 : 0;

        if (dim == shape.size() - 1) {
            out << "[";
            for (size_t i = 0; i < shape[dim]; ++i) {
                if (i > 0) out << ", ";
                out << std::setw(static_cast<int>(max_width)) << data[index++];
            }
            out << "]";
        } else if (shape.size() >= 2 && dim == pre_last) {
            out << "[";
            out << "\n";
            for (size_t i = 0; i < shape[dim]; ++i) {
                out << std::string(indent + 4, ' ');
                print_dim(dim + 1, index, indent + 4);
                if (i != shape[dim] - 1) out << ",";
                out << "\n";
            }
            out << std::string(indent, ' ') << "]";
        } else {
            out << "[";
            if (shape[dim] > 0) {
                out << "\n" << std::string(indent + 4, ' ');
                for (size_t i = 0; i < shape[dim]; ++i) {
                    if (i > 0) out << ", ";
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

void BoundTensor::dump(std::ostream &out) const {
    size_t rank = shape.size();
    out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    out.write(reinterpret_cast<const char*>(shape.data()), static_cast<std::streamsize>(rank * sizeof(size_t)));

    size_t total = num_elements(shape);
    const float* data_ptr = start;
    out.write(reinterpret_cast<const char*>(data_ptr), static_cast<std::streamsize>(total * sizeof(float)));
}

Tensor Graph::load(const std::string &filename, const std::string &name) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Graph::load: cannot open file " + filename);

    // Чтение ранга
    size_t rank;
    in.read(reinterpret_cast<char*>(&rank), sizeof(rank));
    if (!in) throw std::runtime_error("Graph::load: failed to read rank");

    // Чтение размерностей
    std::vector<size_t> shape(rank);
    in.read(reinterpret_cast<char*>(shape.data()), static_cast<std::streamsize>(rank * sizeof(size_t)));
    if (!in) throw std::runtime_error("Graph::load: failed to read shape");

    size_t total = num_elements(shape);

    // Создаём тензор
    Tensor t = allocate(shape, name);
    // Получаем указатель на данные
    float* data = bound.at(t.id);
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(total * sizeof(float)));
    if (!in) throw std::runtime_error("Graph::load: failed to read tensor data");

    return t;
}

const CompatibleInt *Graph::to_size_tensor(const Tensor &t) {
    if (size_tensors.contains(t.id))
        return reinterpret_cast<const CompatibleInt*>(bound.at(t.id));
    auto it = bound.find(t.id);
    if (it == bound.end())
        throw std::runtime_error("Tensor must be initialized before converting to size tensor");
    if (results.count(t.id))
        throw std::runtime_error("Result tensor cannot become a size tensor");

    size_tensors.insert(t.id);
    float* src = it->second;
    size_t nelem = usable.at(src);
    auto* dst = reinterpret_cast<CompatibleInt*>(src);
    for (size_t i = 0; i < nelem; ++i)
        dst[i] = static_cast<CompatibleInt>(src[i]);
    return dst;
}
