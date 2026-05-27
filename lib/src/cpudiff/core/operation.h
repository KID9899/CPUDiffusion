//
// Created by iliya on 5/22/26.
//

#include <cstdint>
#include <unordered_set>

#pragma once

// forward declaration
class UniqueTensor;

enum class OperationId : uint8_t {
    FILL, RANDN,
    VIEW, LIKE,
    INDEX, // INDEX - особенная операция, обрабатывается отдельно и не входит ни в какие группы
    TRANSPOSE,
    REPR, DUMP,
    ADD, SUB, FADD, FSUB,
    MUL, DIV, FMUL, FDIV,
    NEGATE, INVERT,
    MATMUL,
    EXP, RELU, SIGMOID, TANH, POW
};

inline const char* operation_name(OperationId id) {
    using enum OperationId;
    switch (id) {
        case FILL:      return "fill";      case RANDN:  return "randn";
        case VIEW:      return "view";      case LIKE:   return "like";
        case INDEX:     return "";
        case TRANSPOSE: return "transpose";
        case REPR:      return "print";     case DUMP:   return "save";
        case ADD:       return "+";         case SUB:    return "-";
        case FADD:      return "+";         case FSUB:   return "-";
        case MUL:       return "*";         case DIV:    return "/";
        case FMUL:      return "*";         case FDIV:   return "/";
        case NEGATE:    return "* -1";      case INVERT: return "^ -1";
        case MATMUL:    return "@";
        case EXP:       return "e ^";       case RELU:   return "relu";
        case SIGMOID:   return "sigmoid";   case TANH:   return "tanh";
        case POW:       return "^";
        default: return "?";
    }
}

namespace OperationGroups {
    using enum OperationId;
    const std::unordered_set<OperationId> arg2_is_tensor = {
            ADD, SUB,
            MUL, DIV,
            MATMUL
    };
    const std::unordered_set<OperationId> arg2_is_float = {
            FILL,
            FADD, FSUB,
            FMUL, FDIV,
            POW
    };
    const std::unordered_set<OperationId> arg2_is_string = {
            REPR, DUMP
    };
    const std::unordered_set<OperationId> arg2_is_null = {
            VIEW, LIKE,
            RANDN,
            TRANSPOSE,
            NEGATE, INVERT,
            EXP, RELU, SIGMOID, TANH
    };
    const std::unordered_set<OperationId> no_result = {
            FILL, RANDN,
            REPR, DUMP,
    };
    const std::unordered_set<OperationId> may_reuse_args = {
            LIKE,
            ADD, SUB, FADD, FSUB,
            MUL, DIV, FMUL, FDIV,
            NEGATE, INVERT,
            EXP, RELU, SIGMOID, TANH, POW
    };
    const std::unordered_set<OperationId> must_reuse_arg1 = {
            VIEW
    };
}
