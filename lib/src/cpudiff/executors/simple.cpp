//
// Created by iliya on 5/23/26.
//

#include <algorithm>
#include <execution>
#include <fstream>
#include <random>
#include <cmath>
#include <functional>
#include <numeric>

#include "simple.h"

const std::unordered_set<OperationId> &SimpleExecutor::getSupportedOperation() const {
    static std::unordered_set<OperationId> supported = {
            OperationId::FILL, OperationId::RANDN,
            OperationId::TRANSPOSE,
            OperationId::VIEW, OperationId::REPR, OperationId::DUMP,
            OperationId::ADD, OperationId::SUB, OperationId::MUL, OperationId::DIV,
            OperationId::FADD, OperationId::FSUB, OperationId::FMUL, OperationId::FDIV,
            OperationId::NEGATE, OperationId::INVERT,
            OperationId::MATMUL,
            OperationId::EXP, OperationId::RELU, OperationId::SIGMOID, OperationId::TANH
    };
    return supported;
}

void SimpleExecutor::execute() const {
    for (const CompleteOperation &op : operations()) {
        // prefetch обязательных аргументов
        op.arg1.prefetch();

        const auto &shape1 = op.arg1.shape;
        ptrdiff_t n1 = 1;
        for (size_t d : shape1) n1 *= d;
        float *src1 = op.arg1.start;

        switch (op.id) {
            case OperationId::FILL: {
                float val = std::get<float>(op.arg2);
                std::fill(std::execution::par, src1, src1 + n1, val);
                break;
            }
            case OperationId::RANDN: {
                auto gen = []() {
                    thread_local std::mt19937 engine(std::random_device{}());
                    thread_local std::normal_distribution<float> dist(0.0f, 1.0f);
                    return dist(engine);
                };
                std::generate(std::execution::par, src1, src1 + n1, gen);
                break;
            }

            case OperationId::TRANSPOSE: {
                op.result.prefetch();
                float *res = op.result.start;
                const size_t last_dim = shape1.back();
                const size_t pre_last_dim = shape1[shape1.size() - 2];
                const size_t block_size = pre_last_dim * last_dim;
                const size_t num_blocks = n1 / block_size;

                std::vector<size_t> block_ids(num_blocks);
                std::iota(block_ids.begin(), block_ids.end(), 0);
                std::for_each(std::execution::par, block_ids.begin(), block_ids.end(),
                              [&](size_t b) {
                                  size_t base = b * block_size;
                                  for (size_t i = 0; i < pre_last_dim; ++i)
                                      for (size_t j = 0; j < last_dim; ++j)
                                          res[base + j * pre_last_dim + i] = src1[base + i * last_dim + j];
                              });
                break;
            }

            case OperationId::VIEW:
                break;

            case OperationId::REPR:
            case OperationId::DUMP: {
                const char* filename = std::get<const char*>(op.arg2);
                std::ofstream out(filename);
                if (!out) throw std::runtime_error(std::string("Cannot open file: ") + filename);
                if (op.id == OperationId::REPR)
                    op.arg1.repr(out);
                else
                    op.arg1.dump(out);
                if (!out) throw std::runtime_error(std::string("Error writing to file: ") + filename);
                break;
            }

            case OperationId::ADD:
            case OperationId::SUB:
            case OperationId::MUL:
            case OperationId::DIV: {
                const BoundTensor &src2_t = std::get<BoundTensor>(op.arg2);
                src2_t.prefetch();
                op.result.prefetch();
                const auto &shape2 = src2_t.shape;
                ptrdiff_t n2 = 1;
                for (size_t d : shape2) n2 *= d;
                float *src2 = src2_t.start;
                float *res = op.result.start;

                std::function<float(float, float)> f;
                switch (op.id) {
                    case OperationId::ADD: f = [](float x, float y) { return x + y; }; break;
                    case OperationId::SUB: f = [](float x, float y) { return x - y; }; break;
                    case OperationId::MUL: f = [](float x, float y) { return x * y; }; break;
                    case OperationId::DIV: f = [](float x, float y) { return x / y; }; break;
                    default: break;
                }

                for (float *s = src1, *r = res; s < src1 + n1; s += n2, r += n2)
                    std::transform(std::execution::par_unseq,
                                   s, s + n2, src2, r, f);
                break;
            }

            case OperationId::FADD:
            case OperationId::FSUB:
            case OperationId::FMUL:
            case OperationId::FDIV:
            case OperationId::NEGATE:
            case OperationId::INVERT: {
                op.result.prefetch();
                float val = (op.id == OperationId::NEGATE || op.id == OperationId::INVERT) ?
                            0.0f : std::get<float>(op.arg2);
                std::function<float(float)> f;
                switch (op.id) {
                    case OperationId::FADD: f = [val](float x) { return x + val; }; break;
                    case OperationId::FSUB: f = [val](float x) { return x - val; }; break;
                    case OperationId::FMUL: f = [val](float x) { return x * val; }; break;
                    case OperationId::FDIV: f = [val](float x) { return x / val; }; break;
                    case OperationId::NEGATE: f = [](float x) { return -x; }; break;
                    case OperationId::INVERT: f = [](float x) { return 1.0f / x; }; break;
                    default: break;
                }
                std::transform(std::execution::par_unseq,
                               src1, src1 + n1, op.result.start, f);
                break;
            }

            case OperationId::MATMUL: {
                const BoundTensor &src2_t = std::get<BoundTensor>(op.arg2);
                src2_t.prefetch();
                op.result.prefetch();
                const auto &shape2 = src2_t.shape;
                size_t M, N, K;
                if (shape1.size() >= 2 && shape2.size() >= 2) {
                    M = shape1[shape1.size() - 2];
                    N = shape1.back();
                    K = shape2.back();
                } else if (shape2.size() >= 2) {
                    M = 1;
                    N = shape1.back();
                    K = shape2.back();
                } else {
                    M = 1; N = shape1.back(); K = 1;
                }

                ptrdiff_t n2 = 1;
                for (size_t d : shape2) n2 *= d;
                const size_t matrix1_sz = M * N;
                const size_t matrix2_sz = N * K;
                const size_t result_matrix_sz = M * K;
                const size_t num_m1 = n1 / matrix1_sz;
                const size_t num_m2 = n2 / matrix2_sz;
                const size_t mega_blocks = num_m1 / num_m2;

                float *src2 = src2_t.start;
                float *res = op.result.start;

                std::vector<size_t> mega_ids(mega_blocks);
                std::iota(mega_ids.begin(), mega_ids.end(), 0);
                std::for_each(std::execution::par, mega_ids.begin(), mega_ids.end(),
                              [&](size_t mb) {
                                  const float *src1_mega = src1 + mb * num_m2 * matrix1_sz;
                                  float *res_mega = res + mb * num_m2 * result_matrix_sz;
                                  for (size_t i = 0; i < num_m2; ++i) {
                                      const float *A = src1_mega + i * matrix1_sz;
                                      const float *B = src2 + i * matrix2_sz;
                                      float *C = res_mega + i * result_matrix_sz;
                                      for (size_t row = 0; row < M; ++row)
                                          for (size_t col = 0; col < K; ++col) {
                                              float sum = 0.0f;
                                              for (size_t inner = 0; inner < N; ++inner)
                                                  sum += A[row * N + inner] * B[inner * K + col];
                                              C[row * K + col] = sum;
                                          }
                                  }
                              });
                break;
            }

            case OperationId::EXP:
            case OperationId::RELU:
            case OperationId::SIGMOID:
            case OperationId::TANH: {
                op.result.prefetch();
                std::function<float(float)> f;
                switch (op.id) {
                    case OperationId::EXP: f = [](float x) { return std::exp(x); }; break;
                    case OperationId::RELU: f = [](float x) { return x > 0.0f ? x : 0.0f; }; break;
                    case OperationId::SIGMOID: f = [](float x) { return 1.0f / (1.0f + std::exp(-x)); }; break;
                    case OperationId::TANH: f = [](float x) { return std::tanh(x); }; break;
                    default: break;
                }
                std::transform(std::execution::par_unseq,
                               src1, src1 + n1, op.result.start, f);
                break;
            }

            default:
                throw std::runtime_error("Unsupported operation in SimpleExecutor");
        }
    }
}