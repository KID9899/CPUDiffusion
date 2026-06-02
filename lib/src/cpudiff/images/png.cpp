//
// Created by iliya on 5/24/26.
//

#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <sstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "png.h"

// Запись одноканального PNG
static void write_grayscale_png(const std::string& path, size_t w, size_t h, const float* data) {
    std::vector<unsigned char> pixels(w * h);
    for (size_t i = 0; i < w * h; ++i) {
        float v = data[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        pixels[i] = static_cast<unsigned char>(v * 255.0f);
    }
    if (stbi_write_png(path.c_str(), static_cast<int>(w), static_cast<int>(h), 1, pixels.data(), static_cast<int>(w)) == 0) {
        throw std::runtime_error("Failed to write PNG: " + path);
    }
}

// Основная функция
void CpuDiffImages::save_images(const Tensor &t, const std::string& filename) {
    const std::vector<size_t> &shape = t.shape();
    const float *data = t.touch();

    if (!filename.ends_with(".png"))
        throw std::runtime_error("Only PNG format is supported");

    const size_t rank = shape.size();
    if (rank < 2)
        throw std::runtime_error("Tensor must have at least 2 dimensions");
    if (rank > 12)
        throw std::runtime_error("Tensor rank must be <= 12, otherwise %d placeholders overflow");


    const size_t batch_rank = rank - 2;
    const size_t H = shape[batch_rank];
    const size_t W = shape[batch_rank + 1];

    // Проверка имени файла на допустимость шаблонов %d
    for (size_t i = 0; i < filename.size(); ++i) {
        if (filename[i] == '%') {
            if (i + 1 >= filename.size() || !std::isdigit(filename[i+1])) {
                throw std::runtime_error("Invalid placeholder in filename: expected single digit after %");
            }
            int idx = filename[i+1] - '0';
            if (batch_rank == 0) {
                throw std::runtime_error("Placeholder %d is not allowed for 2D tensor");
            }
            if (idx >= batch_rank) {
                throw std::runtime_error("Placeholder %" + std::to_string(idx) +
                                         " is out of range for this tensor (max allowed %" +
                                         std::to_string(batch_rank - 1) + ")");
            }
        }
    }

    // Вычисление страйдов
    std::vector<size_t> stride(rank, 1);
    for (int i = static_cast<int>(rank - 2); i >= 0; --i) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }

    if (batch_rank == 0) {
        // Двумерный случай
        write_grayscale_png(filename, W, H, data);
        return;
    }

    // Перебор всех комбинаций индексов
    std::vector<size_t> index(batch_rank, 0);
    size_t total_slices = 1;
    for (int i = 0; i < batch_rank; ++i) total_slices *= shape[i];

    for (size_t slice = 0; slice < total_slices; ++slice) {
        // Вычисляем многомерный индекс из линейного
        size_t rem = slice;
        for (int i = static_cast<int>(batch_rank) - 1; i >= 0; --i) {
            index[i] = rem % shape[i];
            rem /= shape[i];
        }

        // Формируем имя файла, заменяя %d
        std::string path = filename;
        size_t pos = 0;
        while ((pos = path.find('%', pos)) != std::string::npos) {
            if (pos + 1 < path.size() && std::isdigit(path[pos + 1])) {
                int d = path[pos + 1] - '0';
                path.replace(pos, 2, std::to_string(index[d]));
                // не двигаем pos, так как длина могла измениться
            } else {
                // на всякий случай
                ++pos;
            }
        }

        // Смещение до начала среза
        size_t offset = 0;
        for (int i = 0; i < batch_rank; ++i) {
            offset += index[i] * stride[i];
        }

        write_grayscale_png(path, W, H, data + offset);
    }
}