//
// Created by iliya on 5/20/26.
//

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <nlohmann/json.hpp>

#include "reader.h"

using json = nlohmann::json;

SafeTensorsFile::SafeTensorsFile(const std::string &path, Graph *graph): graph(graph) {
    // 1. Открыть файл
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // 2. Узнать размер файла
    struct stat st;
    if (fstat(fd, &st) == -1) {
        close(fd);
        throw std::runtime_error("Cannot stat file: " + path);
    }
    size_t file_size = static_cast<size_t>(st.st_size);
    if (file_size < 8) {
        close(fd);
        throw std::runtime_error("File too small to be a valid safetensors file");
    }

    // 3. Отобразить файл целиком в память (только на чтение)
    void *mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("mmap failed for file: " + path);
    }

    // Закрываем файл после всего
    close(fd);

    // 4. Прочитать размер JSON-заголовка (первые 8 байт)
    uint64_t header_size;
    std::memcpy(&header_size, mapped, sizeof(header_size));
    // safetensors использует little-endian, поэтому предполагаем, что платформа little-endian
    if (header_size > file_size - 8) {
        munmap(mapped, file_size);
        throw std::runtime_error("Header size exceeds file size");
    }

    // 5. Извлечь JSON-строку (начинается сразу после 8 байт размера)
    const char *json_start = static_cast<const char*>(mapped) + 8;
    std::string json_str(json_start, header_size);

    // 6. Парсим JSON
    json root = json::parse(json_str);
    // Там что-то типа {"tensor_name": {"dtype": string, "shape": [size..], "data_offsets": [uint64, uint64]}...}
    if (!root.is_object()) {
        munmap(mapped, file_size);
        throw std::runtime_error("Invalid safetensors header: top-level must be an object");
    }

    // Смещение в файле, откуда начинаются данные тензоров
    const size_t data_section_offset = 8 + header_size;
    // Указатель на начало секции данных внутри отображения
    char *data_base = static_cast<char*>(mapped) + data_section_offset;

    // 7. Проходимся по тензорам
    for (auto &[tensor_name, tensor_info] : root.items()) {
        if (!tensor_name.empty() && tensor_name == "__metadata__") continue;
        if (!tensor_info.contains("dtype") ||
            !tensor_info.contains("shape") ||
            !tensor_info.contains("data_offsets")) {
            munmap(mapped, file_size);
            throw std::runtime_error("Missing fields in tensor info for: " + tensor_name);
        }

        std::string dtype_str = tensor_info["dtype"].get<std::string>();
        Dtype dtype = dtype_from_string(dtype_str);

        std::vector<uint64_t> shape_uint64 = tensor_info["shape"].get<std::vector<uint64_t>>();
        std::vector<size_t> shape;
        for (const auto &i: shape_uint64) {
            if (!std::cmp_less_equal(i, SIZE_MAX)) throw std::overflow_error("The tensor data size overflows the size_t type");
            shape.push_back(i);
        }

        auto offsets = tensor_info["data_offsets"].get<std::vector<uint64_t>>();
        if (offsets.size() != 2) {
            munmap(mapped, file_size);
            throw std::runtime_error("data_offsets must be an array of two integers");
        }
        uint64_t start_off = offsets[0];
        uint64_t end_off   = offsets[1];
        if (start_off > end_off || end_off > file_size - data_section_offset) {
            munmap(mapped, file_size);
            throw std::runtime_error("Invalid data offsets for tensor: " + tensor_name);
        }

        // Заполняем TensorHeader
        Tensor th = graph->add_existing(static_cast<void*>(data_base + start_off), static_cast<void*>(data_base + end_off), std::move(shape), tensor_name, dtype);

        // Сохраняем в наш класс
        size_t idx = tensors_.size();
        tensors_.push_back(th);
        names_.push_back(tensor_name);
        name_to_idx_[tensor_name] = idx;
    }

    // Сохраняем указатель и размер маппинга
    mmap_ptr_  = mapped;
    mmap_size_ = file_size;
}

SafeTensorsFile::~SafeTensorsFile() {
    if (mmap_ptr_ != nullptr && mmap_ptr_ != MAP_FAILED) {
        munmap(mmap_ptr_, mmap_size_);
    }
}
