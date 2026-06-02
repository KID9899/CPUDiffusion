//
// Created by iliya on 5/22/26.
//

#include "module.h"

#pragma once


class ModuleClass(ReLU) {
public:
    static Tensor forward(const Tensor &x);
};

class ModuleClass(Sigmoid) {
public:
    static Tensor forward(const Tensor &x);
};

class ModuleClass(Flatten) {
public:
    static Tensor forward(const Tensor &x);
};

class ModuleClass(Tanh) {
public:
    static Tensor forward(const Tensor &x);
};

class ModuleClass(GELU) {
public:
    static Tensor forward(const Tensor &x);
};
