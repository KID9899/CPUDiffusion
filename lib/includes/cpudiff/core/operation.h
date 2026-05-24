//
// Created by iliya on 5/22/26.
//

#include <cstdint>

#pragma once

// forward declaration
class UniqueTensor;

enum class OperationId : uint8_t {
    FILL, RANDN,
    VIEW, COPY, TRANSPOSE,
    REPR, DUMP,
    ADD, SUB, NEGATE,
    MUL, DIV, INVERT, FMUL, FDIV,
    MATMUL,
    EXP, RELU, SIGMOID
};

struct GraphOperation {
    union SecondArg { const UniqueTensor *t; float f; const char *s; };

    OperationId id;
    const UniqueTensor *src1;
    SecondArg src2;
    UniqueTensor *result;

    inline GraphOperation(OperationId id, const UniqueTensor *src1, SecondArg src2, UniqueTensor *result)
        : id(id), src1(src1), src2(src2), result(result) {}
};

inline const char* operation_name(OperationId id) {
    switch (id) {
        case OperationId::VIEW:     return "view";
        case OperationId::REPR:     return "print";
        case OperationId::DUMP:     return "save";
        case OperationId::ADD:      return "+";
        case OperationId::SUB:      return "-";
        case OperationId::MUL:      return "*";
        case OperationId::DIV:      return "/";
        case OperationId::FMUL:     return "*";
        case OperationId::FDIV:     return "/";
        case OperationId::MATMUL:   return "@";
        case OperationId::NEGATE:   return "-";
        case OperationId::INVERT:   return "^-1";
        case OperationId::FILL:     return "fill";
        case OperationId::RANDN:    return "randn";
        case OperationId::COPY:     return "copy";
        case OperationId::EXP:      return "Exp";
        case OperationId::RELU:     return "ReLU";
        case OperationId::SIGMOID:  return "Sigmoid";
        case OperationId::TRANSPOSE:return "T";
        default: return "?";
    }
}
