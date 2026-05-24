//
// Created by iliya on 5/23/26.
//

#include "cpudiff/core.h"

#pragma once


class SimpleExecutor final: public GraphExecutor {
protected:
    virtual const std::unordered_set<OperationId> &getSupportedOperation() const override;
public:
    inline SimpleExecutor(Graph *graph): GraphExecutor(graph) {}
    virtual void execute() const;
};
