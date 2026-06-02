//
// Created by iliya on 5/21/26.
//

#include <unordered_map>
#include <stdexcept>
#include <string>
#include <concepts>
#include <algorithm>
#include <sstream>

#include "cpudiff/core.h"
#include "cpudiff/safetensors.h"

#pragma once

// Интерфейс, предоставляющий общее взаимодействие с любыми модулями
class Module {
private:
    inline static constinit unsigned int MODULE_ID = 0;
    std::unordered_map<std::string, Module*> children;
    std::unordered_map<std::string, Tensor*> weights;
    std::unordered_map<std::string, IndexList*> indexes;
    void load(const SafeTensorsFile *file, const std::string &prefix);
    void set_module_name() { module_name = CLASS_NAME() + std::string("#") + std::to_string(++MODULE_ID);}
protected:
    Graph *graph;
    std::string module_name;

    // Регистрация подмодулей и параметров
    void register_module(const std::string &name, Module &module);
    void register_module(const std::string &name, Module *module);
    void register_tensor(const std::string &name, Tensor &tensor);
    void register_indexes(const std::string &name, IndexList &list);

    // Название класса
    virtual const char *CLASS_NAME() { return "Module"; };

    // Вызывается после загрузки весов
    virtual void on_load() {};

    // Низкоуровневая функция, которую предоставляет интерфейс Module
    virtual AnyTensor forward(const AnyTensor &x) = 0;
public:
    explicit Module(Graph *graph);
    Module(): Module(Graph::get_active()) {}

    // Загрузить веса и рекурсивно подмодули
    void load(const SafeTensorsFile *file);

    // Вызов модуля как функции - единственный способ работы извне
    template<typename... Args>
    inline AnyTensor call(Args... args) {
        return forward({args...});
    }
    template<typename... Args>
    inline AnyTensor operator()(Args... args) {
        return call(args...);
    }

};


// Определение структуры для получения информации о функции
template<typename T>
struct function_traits;

template <typename Ret, typename... Args>
struct function_traits<Ret(Args...)> {
    static constexpr size_t arity = sizeof...(Args);
    using argument_types = std::tuple<Args...>;
    using return_type = Ret;
};

// О методе класса
template<typename Ret, typename Class, typename... Args>
struct function_traits<Ret(Class::*)(Args...)> {
    static constexpr size_t arity = sizeof...(Args);
    using argument_types = std::tuple<Args...>;
};

// И о константной функции тоже
template<typename Ret, typename Class, typename... Args>
struct function_traits<Ret(Class::*)(Args...) const> : function_traits<Ret(Class::*)(Args...)> {};

// И о статической функции
template <typename Ret, typename... Args>
struct function_traits<Ret(*)(Args...)> : function_traits<Ret(Args...)> {};

// Структура для удобного Template
template<std::size_t N>
struct FixedString {
    char data[N] = {};
    constexpr FixedString(const char (&str)[N]) noexcept {
        std::copy_n(str, N, data);
    }
};

// Получение имени типа
template<typename T>
std::string type_name() {
    return typeid(T).name();
}

// Рекурсивно собираем строку из имён типов кортежа
template<typename Tuple, std::size_t... I>
std::string tuple_type_names_impl(std::index_sequence<I...>) {
    std::string result;
    // fold expression (C++17)
    ((result += (I == 0 ? "" : ", ") + type_name<std::tuple_element_t<I, Tuple>>()), ...);
    return result;
}

template<typename Tuple>
std::string tuple_type_names() {
    return tuple_type_names_impl<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

// Удобное использование для наследования:
// class ModuleClass(MyClass) { ... }
#define ModuleClass(name) name final: public BaseModule<name, #name>


template<typename Derived, FixedString ClassName>
class BaseModule: public Module {
private:
    template<typename = void, size_t... I>
    AnyTensor callDerivedForward(const AnyTensor& list, std::index_sequence<I...>) {
        using ArgsTuple = typename function_traits<decltype(&Derived::forward)>::argument_types;
        return dynamic_cast<Derived*>(this)->forward(
                std::tuple_element_t<I, ArgsTuple>(list[I])...
        );
    }
protected:
    const char *CLASS_NAME() override { return ClassName.data; };

    AnyTensor forward(const AnyTensor &x) override {
        using Traits = function_traits<decltype(&Derived::forward)>;
        constexpr size_t expected = Traits::arity;
        if (x.size() != expected) {
            std::ostringstream msg;
            msg << "Module '" << CLASS_NAME() << "' forward(): "
                << "expected " << expected << " argument(s) of type(s) ("
                << tuple_type_names<typename Traits::argument_types>()
                << "), but received " << x.size() << " argument(s)";
            // Если AnyTensor умеет сообщать свой тип, можно добавить:
             msg << " of type(s) (";
             for (size_t i = 0; i < x.size(); ++i) {
                 if (i) msg << ", ";
                 msg << x[i].type_name();
             }
             msg << ")";
            throw std::runtime_error(msg.str());
        }

        return callDerivedForward<>(x, std::make_index_sequence<expected>());
    }
public:
    BaseModule() {
        static_assert(std::is_base_of_v<BaseModule<Derived, ClassName>, Derived>, "Derived must be a subclass of Module<Derived>");
    }
};
