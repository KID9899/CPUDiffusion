//
// Created by iliya on 5/23/26.
//

#include <algorithm>
#include <execution>
#include <fstream>
#include <random>
#include <cmath>

#include "simple.h"

const std::unordered_set<OperationId> &SimpleExecutor::getSupportedOperation() const {
    static std::unordered_set<OperationId> supported = {
            OperationId::FILL, OperationId::RANDN,
            OperationId::COPY, OperationId::TRANSPOSE,
            OperationId::VIEW, OperationId::REPR, OperationId::DUMP,
            OperationId::ADD, OperationId::SUB, OperationId::MUL, OperationId::DIV,
            OperationId::FMUL, OperationId::FDIV, OperationId::NEGATE, OperationId::INVERT,
            OperationId::MATMUL,
            OperationId::EXP, OperationId::RELU, OperationId::SIGMOID
    };
    return supported;
}

void SimpleExecutor::execute() const {
    for (const GraphOperation &current_op: operations()) {
        current_op.src1->prefetch();
        const auto &src1_shape = current_op.src1->shape();

        ptrdiff_t src1_num_elements = 1;
        for (size_t i = 0; i < src1_shape.size(); ++i) src1_num_elements *= src1_shape[i];

        float *src1_start = static_cast<float*>(current_op.src1->data());
        float *src1_end = static_cast<float*>(src1_start) + src1_num_elements;

        switch (current_op.id) {
            case (OperationId::FILL): {
                std::fill(std::execution::par, src1_start, src1_end, current_op.src2.f);
                break;
            }
            case (OperationId::RANDN): {
                auto gen = []() {
                    thread_local std::mt19937 engine(std::random_device{}());
                    thread_local std::normal_distribution<float> dist(0.0f, 1.0f);
                    return dist(engine);
                };
                std::generate(std::execution::par, src1_start, src1_end, gen);
                break;
            }

            case (OperationId::COPY): {
                current_op.result->prefetch();
                float *res = static_cast<float*>(current_op.result->data());
                std::copy(src1_start, src1_end, res);
                break;
            }
            case (OperationId::TRANSPOSE): {
                current_op.result->prefetch();
                float *res = static_cast<float*>(current_op.result->data());

                const size_t last_dim = src1_shape.back();
                const size_t pre_last_dim = src1_shape[src1_shape.size() - 2];
                const size_t block_size = pre_last_dim * last_dim;
                const size_t num_blocks = src1_num_elements / block_size;

                // Параллельная обработка всех блоков
                std::vector<size_t> block_ids(num_blocks);
                std::iota(block_ids.begin(), block_ids.end(), 0);
                std::for_each(std::execution::par, block_ids.begin(), block_ids.end(),
                              [&](size_t b) {
                                  size_t base = b * block_size;
                                  for (size_t i = 0; i < pre_last_dim; ++i) {
                                      for (size_t j = 0; j < last_dim; ++j) {
                                          // (i,j) -> (j,i)
                                          res[base + j * pre_last_dim + i] = src1_start[base + i * last_dim + j];
                                      }
                                  }
                              }
                );
                break;
            }

            case (OperationId::VIEW): {
                break;
            }
            case (OperationId::REPR):
            case (OperationId::DUMP): {
                std::ofstream out(current_op.src2.s);
                if (!out) throw std::runtime_error(std::string("Cannot open file: ") + current_op.src2.s);
                if (current_op.id == OperationId::REPR)
                    current_op.src1->repr(out);
                else
                    current_op.src1->dump(out);
                if (!out) throw std::runtime_error(std::string("Error writing to file: ") + current_op.src2.s);
                break;
            }

            case (OperationId::ADD):
            case (OperationId::SUB):
            case (OperationId::MUL):
            case (OperationId::DIV): {
                current_op.src2.t->prefetch();
                current_op.result->prefetch();

                const auto &src2_shape = current_op.src2.t->shape();

                ptrdiff_t src2_num_elements = 1;
                for (size_t i = 0; i < src2_shape.size(); ++i) src2_num_elements *= src2_shape[i];

                float *src2_start = static_cast<float*>(current_op.src2.t->data());
                float *src2_end = static_cast<float*>(src2_start) + src2_num_elements;

                std::function<float(float, float)> f;
                switch (current_op.id) {
                    case (OperationId::ADD): {
                        f = [](float x, float y) { return x + y; };
                        break;
                    }
                    case (OperationId::SUB): {
                        f = [](float x, float y) { return x - y; };
                        break;
                    }
                    case (OperationId::MUL): {
                        f = [](float x, float y) { return x * y; };
                        break;
                    }
                    case (OperationId::DIV): {
                        f = [](float x, float y) { return x / y; };
                        break;
                    }
                }

                for (
                        float *src = src1_start, *res = static_cast<float*>(current_op.result->data());
                        src < src1_end;
                        src += src2_num_elements, res += src2_num_elements
                ) {
                    std::transform(std::execution::par_unseq,
                                   src, src + src2_num_elements,
                                   src2_start, res, f);
                }
                break;
            }

            case (OperationId::FMUL):
            case (OperationId::FDIV):
            case (OperationId::NEGATE):
            case (OperationId::INVERT): {
                std::function<float(float)> f;
                const float src2 = current_op.src2.f;
                switch (current_op.id) {
                    case (OperationId::FMUL): {
                        f = [src2](float x) { return x * src2; };
                        break;
                    }
                    case (OperationId::FDIV): {
                        f = [src2](float x) { return x / src2; };
                        break;
                    }
                    case (OperationId::NEGATE): {
                        f = [](float x) { return -x; };
                        break;
                    }
                    case (OperationId::INVERT): {
                        f = [](float x) { return 1.f/x; };
                        break;
                    }
                }
                std::transform(std::execution::par_unseq, src1_start, src1_end, static_cast<float*>(current_op.result->data()), f);
                break;
            }

            case (OperationId::MATMUL): {
                current_op.src2.t->prefetch();
                current_op.result->prefetch();
                const auto &src2_shape = current_op.src2.t->shape();
                size_t M, N, K;
                // Перемножение матриц M*N и N*K
                if (src1_shape.size() >= 2 && src2_shape.size() >= 2) {
                    M = src1_shape[src1_shape.size() - 2];
                    N = src1_shape.back();
                    K = src2_shape.back();
                } else if (src2_shape.size() >= 2) {
                    M = 1;
                    N = src1_shape.back();
                    K = src2_shape.back();
                } else {
                    M = 1;
                    N = src1_shape.back();
                    K = 1;
                }

                // Число элементов и число матриц
                ptrdiff_t src2_num_elements = 1;
                for (size_t i = 0; i < src2_shape.size(); ++i)
                    src2_num_elements *= src2_shape[i];

                const size_t matrix1_size = M * N;
                const size_t matrix2_size = N * K;
                const size_t result_matrix_size = M * K;

                const size_t num_matrices_src1 = src1_num_elements / matrix1_size;
                const size_t num_matrices_src2 = src2_num_elements / matrix2_size;

                // Циклически повторяем матрицы в src2
                const size_t mega_blocks = num_matrices_src1 / num_matrices_src2;

                float *src2_start = static_cast<float*>(current_op.src2.t->data());
                float *res_start = static_cast<float*>(current_op.result->data());

                // Параллельная обработка групп матриц
                std::vector<size_t> mega_ids(mega_blocks);
                std::iota(mega_ids.begin(), mega_ids.end(), 0);
                std::for_each(std::execution::par, mega_ids.begin(), mega_ids.end(),
                              [&](size_t mb) {
                                  const float *src1_mega = src1_start + mb * num_matrices_src2 * matrix1_size;
                                  float *res_mega   = res_start   + mb * num_matrices_src2 * result_matrix_size;

                                  for (size_t i = 0; i < num_matrices_src2; ++i) {
                                      const float *A = src1_mega + i * matrix1_size;
                                      const float *B = src2_start + i * matrix2_size;  // src2 всегда один и тот же набор
                                      float *C       = res_mega   + i * result_matrix_size;

                                      for (size_t row = 0; row < M; ++row) {
                                          for (size_t col = 0; col < K; ++col) {
                                              float sum = 0.0f;
                                              for (size_t inner = 0; inner < N; ++inner) {
                                                  sum += A[row * N + inner] * B[inner * K + col];
                                              }
                                              C[row * K + col] = sum;
                                          }
                                      }
                                  }
                              }
                );
                break;
            }

            case (OperationId::EXP):
            case (OperationId::RELU):
            case (OperationId::SIGMOID): {
                current_op.result->prefetch();
                std::function<float(float)> f;
                switch (current_op.id) {
                    case (OperationId::EXP): {
                        f = [](float x) { return std::exp(x); };
                        break;
                    }
                    case (OperationId::RELU): {
                        f = [](float x) { return x > 0.0f ? x : 0.0f; };
                        break;
                    }
                    case (OperationId::SIGMOID): {
                        f = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
                        break;
                    }
                }
                std::transform(std::execution::par_unseq,
                               src1_start, src1_end,
                               static_cast<float*>(current_op.result->data()),
                               f);
                break;
            }
        }
    }
}
