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

static size_t total_elements(const std::vector<size_t> &shape) {
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

Tensor Graph::add_existing(void *start, void *end, const std::vector<size_t> &shape, const std::string &name, Dtype dtype) {
    if (shape.empty()) throw std::runtime_error("It is impossible to create a tensor with an empty array of dimensions");
    if (dtype == FloatDtype) {
        tensors.push_back(std::make_unique<UniqueTensor>(this, shape, name));
        tensors.back()->set_memory(start, end);
        return tensors.back()->reference();
    } else {
        Tensor t = allocate(shape);
        dtype_to_float(static_cast<float*>(t.data()), start, total_elements(shape), dtype);
        return t;
    }
}

Tensor Graph::allocate(const std::vector<size_t> &shape, const std::string &name) {
    if (shape.empty()) throw std::runtime_error("It is impossible to create a tensor with an empty array of dimensions");
    Tensor t = future(name);
    alloc_promised(t.unique, shape);
    return t;
}

void Graph::add_operation(OperationId id, const UniqueTensor *src1, GraphOperation::SecondArg src2, UniqueTensor *result) {
    operations.emplace_back(id, src1, src2, result);
}

Tensor Graph::future(const std::string &name) {
    tensors.push_back(std::make_unique<UniqueTensor>(this, std::vector<size_t>(), name));
    return tensors.back()->reference();
}

void Graph::alloc_promised(UniqueTensor *tensor, const std::vector<size_t> &shape) {
    if (shape.empty()) throw std::runtime_error("It is impossible to create a tensor with an empty array of dimensions");
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

void Graph::dump_dot(const std::string &filename, const std::string &name) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Заголовок DOT-файла
    ofs << "digraph G {\n";
    ofs << "  rankdir=TB;\n";
    ofs << "  node [fontname=\"" << name << "\"];\n\n";

    // -------- Уникальные тензоры --------
    struct UniqueTensorInfo {
        std::string id;
        std::string name;
        std::string dims;
    };

    // Группируем все тензоры по их указателю на память
    std::unordered_map<void*, std::vector<const UniqueTensor*>> data_to_tensors;
    for (const auto& t : tensors) {
        if (t->data() == nullptr) continue;
        data_to_tensors[t->data()].push_back(t.get());
    }

    // Для каждого уникального тензора заполняем информацию и решаем, выносить ли в кластер
    std::unordered_map<const UniqueTensor*, UniqueTensorInfo> utensor_info;

    // Сначала формируем id и подписи для каждого тензора
    for (const auto& [data_ptr, utensors] : data_to_tensors) {
        for (const auto* ut : utensors) {
            UniqueTensorInfo info;
            info.id = "utensor_" + std::to_string(reinterpret_cast<uintptr_t>(ut));

            // Если у тензора есть имя, берём его, иначе используем адрес данных
            if (ut->name().empty()) {
                std::stringstream ss;
                ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(data_ptr);
                info.name = ss.str();
            } else {
                info.name = ut->name();
            }

            // Размерность
            info.dims = "[";
            for (size_t i = 0; i < ut->shape().size(); ++i) {
                if (i != 0) info.dims += ", ";
                info.dims += std::to_string(ut->shape()[i]);
            }
            info.dims += "]";

            utensor_info[ut] = info;
        }
    }

    // -------- Отрисовка групп и одиночных узлов --------
    for (const auto& [data_ptr, utensors] : data_to_tensors) {
        if (utensors.size() > 1) {
            // Кластер с пунктирной рамкой
            ofs << "  subgraph cluster_" << std::hex << reinterpret_cast<uintptr_t>(data_ptr) << " {\n";
            ofs << "    style=solid;\n";
            ofs << "    color=red;\n";
            for (const auto* ut : utensors) {
                const auto& info = utensor_info[ut];
                ofs << "    " << info.id
                    << " [shape=box, style=dashed, color=red, fontcolor=red, label=\""
                    << info.name << "\\n" << info.dims << "\"];\n";
            }
            ofs << "  }\n";
        } else {
            // Один тензор - просто узел без кластера
            const auto* ut = utensors[0];
            const auto& info = utensor_info[ut];
            ofs << "  " << info.id
                << " [shape=box, style=solid, color=red, fontcolor=red, label=\""
                << info.name << "\\n" << info.dims << "\"];\n";
        }
    }
    ofs << "\n";

    // -------- Операции и узлы-ссылки --------
    std::unordered_map<const UniqueTensor*, int> ref_counters;    // счётчики на каждый уникальный тензор
    std::vector<std::tuple<std::string, std::string>> references; // пунктирные рёбра
    std::vector<std::tuple<std::string, std::string>> edges;      // нормальные рёбра

    // Лямбда, создающая (или возвращающая существующую) ссылку на уникальный тензор
    auto get_or_create_ref = [&](const UniqueTensor* t, bool get_new) -> std::string {
        if (!t || !t->data()) return "";

        auto it_info = utensor_info.find(t);
        if (it_info == utensor_info.end()) {
            throw std::runtime_error("A reference to a tensor that does not belong to a graph");
        }
        const std::string& master_id = it_info->second.id;

        // Переиспользуем ссылку, если тензор не менялся
        if (ref_counters.find(t) == ref_counters.end()) {
            ref_counters[t] = 0;
        } else if (!get_new) {
            int count = ref_counters[t] - 1;
            return "ref_" + std::to_string(reinterpret_cast<uintptr_t>(t))
                   + "_" + std::to_string(count);
        }

        int count = ref_counters[t]++;
        std::string ref_id = "ref_" + std::to_string(reinterpret_cast<uintptr_t>(t))
                             + "_" + std::to_string(count);

        // Узел-ссылка
        ofs << "  " << ref_id << " [shape=box, label=\""
            << it_info->second.name << "\\n" << it_info->second.dims << "\"];\n";

        // Пунктир от уникального тензора к ссылке
        references.emplace_back(master_id, ref_id);
        return ref_id;
    };

    size_t op_idx = 0;
    for (const auto& op : operations) {
        std::string op_id = "op_" + std::to_string(op_idx++);
        std::string op_label = operation_name(op.id);
        if (op.id == OperationId::FMUL || op.id == OperationId::FDIV)
            op_label += std::to_string(op.src2.f);

        ofs << "  " << op_id << " [shape=diamond, label=\"" << op_label << "\"];\n";

        // Первый операнд
        if (op.src1) {
            std::string ref = get_or_create_ref(op.src1, false);
            if (!ref.empty()) edges.emplace_back(ref, op_id);
        }

        // Второй операнд
        if (op.id == OperationId::ADD || op.id == OperationId::SUB ||
            op.id == OperationId::MUL || op.id == OperationId::DIV ||
            op.id == OperationId::MATMUL) {
            if (op.src2.t) {
                std::string ref = get_or_create_ref(op.src2.t, false);
                if (!ref.empty()) edges.emplace_back(ref, op_id);
            }
        }

        // Результат
        if (op.result) {
            std::string ref = get_or_create_ref(op.result, true);
            if (!ref.empty()) edges.emplace_back(op_id, ref);
        }
    }

    // Отрисовываем пунктирные рёбра
    for (const auto& [from, to] : references) {
        ofs << "  " << from << " -> " << to << " [style=dashed, color=red];\n";
    }
    // Отрисовываем обычные рёбра
    for (const auto& [from, to] : edges) {
        ofs << "  " << from << " -> " << to << ";\n";
    }

    ofs << "}\n";
    ofs.close();
}

Tensor Graph::link(float* data, const std::vector<size_t>& shape, const std::string& name) {
    if (shape.empty()) throw std::runtime_error("It is impossible to create a tensor with an empty array of dimensions");
    return add_existing(
            static_cast<void*>(data), static_cast<void*>(data + static_cast<ptrdiff_t>(total_elements(shape))), shape, name
    );
}

Tensor Graph::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Graph::load: cannot open file " + filename);
    }

    // Чтение имени
    std::string name;
    char ch;
    while (in.get(ch) && ch != '\0') {
        name.push_back(ch);
    }
    if (!in) {
        throw std::runtime_error("Graph::load: unexpected end of file while reading tensor name");
    }

    // Чтение ранга
    size_t rank;
    in.read(reinterpret_cast<char*>(&rank), sizeof(rank));
    if (!in) {
        throw std::runtime_error("Graph::load: failed to read rank");
    }

    // Чтение размерностей
    std::vector<size_t> shape(rank);
    in.read(reinterpret_cast<char*>(shape.data()), rank * sizeof(size_t));
    if (!in) {
        throw std::runtime_error("Graph::load: failed to read shape");
    }

    size_t num_elements = total_elements(shape);
    Tensor t = allocate(shape, name);
    float* data = static_cast<float*>(t.data());
    in.read(reinterpret_cast<char*>(data), num_elements * sizeof(float));
    if (!in) throw std::runtime_error("Graph::load: failed to read tensor data");

    return t;
}
