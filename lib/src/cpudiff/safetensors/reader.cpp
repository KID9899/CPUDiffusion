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

SafeTensorsFile::SafeTensorsFile(const std::string &path, Graph *graph) : graph(graph) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("Cannot open file: " + path);

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

    void *mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) throw std::runtime_error("mmap failed for file: " + path);

    uint64_t header_size;
    std::memcpy(&header_size, mapped, sizeof(header_size));
    if (header_size > file_size - 8) {
        munmap(mapped, file_size);
        throw std::runtime_error("Header size exceeds file size");
    }

    const char *json_start = static_cast<const char*>(mapped) + 8;
    std::string json_str(json_start, header_size);

    json root = json::parse(json_str);
    if (!root.is_object()) {
        munmap(mapped, file_size);
        throw std::runtime_error("Invalid safetensors header");
    }

    const size_t data_section_offset = 8 + header_size;
    const char *data_base = static_cast<const char*>(mapped) + data_section_offset;

    // Обход тензоров
    for (auto &[tensor_name, tensor_info] : root.items()) {
        if (tensor_name == "__metadata__") continue;

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
        for (auto s : shape_uint64) shape.push_back(static_cast<size_t>(s));

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

        const void *src_ptr = data_base + start_off;
        size_t byte_size = static_cast<size_t>(end_off - start_off);
        size_t nelem = 1;
        for (auto d : shape) nelem *= d;

        Tensor tensor;
        if (dtype == FloatDtype) {
            // Выделяем память в графе и копируем данные
            tensor = graph->allocate(shape, tensor_name);
            float *dst = graph->force_bind(tensor, false);
            // Проверка размера
            if (byte_size != nelem * sizeof(float)) {
                munmap(mapped, file_size);
                throw std::runtime_error("Data size mismatch for float tensor: " + tensor_name);
            }
            std::memcpy(dst, src_ptr, byte_size);
        } else {
            // Преобразование типа
            tensor = graph->allocate(shape, tensor_name);
            float *dst = graph->force_bind(tensor);
            dtype_to_float(dst, src_ptr, nelem, dtype);
        }

        size_t idx = tensors_.size();
        tensors_.push_back(tensor);
        names_.push_back(tensor_name);
        name_to_idx_[tensor_name] = idx;
    }

    munmap(mapped, file_size);
}
