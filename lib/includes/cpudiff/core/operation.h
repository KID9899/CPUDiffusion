//
// Created by iliya on 5/22/26.
//

#include <cstdint>
#include <unordered_set>

#pragma once

enum class OperationId : uint8_t {
    FILL, RANDN,
    VIEW, LIKE, // При исполнении можно игнорировать
    GATHER, // Особенные операции, обрабатываются отдельно и не входят ни в какие группы
    TRANSPOSE,
    REPR, DUMP,
    ADD, SUB, FADD, FSUB,
    MUL, DIV, FMUL, FDIV,
    NEGATE, INVERT,
    MATMUL,
    EXP, RELU, SIGMOID, TANH, POW,
    REDUCE_MIN, REDUCE_MAX, REDUCE_SUM, REDUCE_MEAN,
    REPEAT
};

inline const char* operation_name(OperationId id) {
    using enum OperationId;
    switch (id) {
        case FILL:      return "fill";      case RANDN:  return "randn";
        case VIEW:      return "view";      case LIKE:   return "like";
        case GATHER:    return "gather";
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

        case REDUCE_MIN:    return "min";  case REDUCE_MAX:  return "max";
        case REDUCE_SUM:    return "sum";  case REDUCE_MEAN: return "mean";
        case REPEAT:        return "repeat";
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
            EXP, RELU, SIGMOID, TANH, POW,
            REDUCE_MIN, REDUCE_MAX, REDUCE_SUM, REDUCE_MEAN,
    };
    const std::unordered_set<OperationId> must_reuse_arg1 = {
            VIEW
    };
    inline const std::unordered_set<OperationId> arg2_is_indexlist = {
            GATHER
    };
    const std::unordered_set<OperationId> arg2_is_size = {
            REDUCE_MIN, REDUCE_MAX, REDUCE_SUM, REDUCE_MEAN,
            REPEAT
    };
}
