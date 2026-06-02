//
// Created by iliya on 5/29/26.
//

#include "tensor.h"

#pragma once

// Определение типа для полиморфизма модулей
class AnyTensor {
private:
    using Val = std::variant<Tensor, IndexList, float>;
    using Vec = std::vector<Val>;
    std::variant<Vec, Val> data;

public:
    template <typename... Args>
    requires std::constructible_from<Val, Args...>
    AnyTensor(const Val &arg1, const Val &arg2, Args&&... other_args): data(Vec{
        arg1, arg2, Val(std::forward<Args>(other_args)) ...
    }) {}
    AnyTensor(const Val &arg): data(arg) {}
    AnyTensor(const Tensor &arg): data(Val(arg)) {}
    AnyTensor(const IndexList &arg): data(Val(arg)) {}
    AnyTensor(float arg): data(Val(arg)) {}

    inline operator Val() const {
        if (const auto &c = std::get_if<Val>(&data))
            return *c;
        const auto &vec = std::get<Vec>(data);
        if (vec.size() != 1) throw std::runtime_error("It is impossible to write more than one value to Tensor");
        return vec[0];
    }
    inline operator Tensor() const { return operator const Tensor&(); }
    inline explicit operator Tensor&() {
        if (const auto &c = std::get_if<Val>(&data))
            return std::get<Tensor>(*c);
        auto &vec = std::get<Vec>(data);
        if (vec.size() != 1) throw std::runtime_error("It is impossible to write more than one value to Tensor");
        return std::get<Tensor>(vec[0]);
    }
    inline explicit operator const Tensor&() const {
        if (const auto &c = std::get_if<Val>(&data))
            return std::get<Tensor>(*c);
        const auto &vec = std::get<Vec>(data);
        if (vec.size() != 1) throw std::runtime_error("It is impossible to write more than one value to Tensor");
        return std::get<Tensor>(vec[0]);
    }
    inline operator IndexList() {
        if (const auto &c = std::get_if<Val>(&data))
            return std::get<IndexList>(*c);
        const auto &vec = std::get<Vec>(data);
        if (vec.size() != 1) throw std::runtime_error("It is impossible to write more than one value to IndexList");
        return std::get<IndexList>(vec[0]);
    }
    inline explicit operator float() {
        if (const auto &c = std::get_if<Val>(&data))
            return std::get<float>(*c);
        const auto &vec = std::get<Vec>(data);
        if (vec.size() != 1) throw std::runtime_error("It is impossible to write more than one value to float");
        return std::get<float>(vec[0]);
    }

    template<typename... Types>
    std::tuple<Types...> get() const {
        constexpr size_t N = sizeof...(Types);

        // data содержит вектор
        if (const auto* vec = std::get_if<Vec>(&data)) {
            if (vec->size() != N) {
                throw std::out_of_range(
                        "Number of requested values (" + std::to_string(N) +
                        ") != number of available values (" + std::to_string(vec->size()) + ")"
                );
            }

            auto extract = [&](size_t idx, auto tag) -> decltype(auto) {
                using T = typename decltype(tag)::type;
                return std::get<T>((*vec)[idx]);
            };

            return [&extract]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::tuple<Types...>{ extract(Is, std::type_identity<Types>{})... };
            }(std::index_sequence_for<Types...>{});
        }

        // data содержит одиночное значение
        if constexpr (N != 1) {
            throw std::out_of_range("Cannot extract multiple values from a single value");
        } else {
            using T0 = std::tuple_element_t<0, std::tuple<Types...>>;
            return std::tuple<T0>{ this->operator T0() };
        }
    }

    inline AnyTensor operator[](size_t idx) const {
        if (const auto &c = std::get_if<Vec>(&data))
            return std::get<Vec>(data)[idx];
        if (idx == 0)
            return std::get<Val>(data);
        else
            throw std::out_of_range("Index out of range");
    }
    inline std::string type_name() const {
        if (const auto &c = std::get_if<Val>(&data))
            return std::visit([](const auto& arg) {
                return std::string(typeid(arg).name());
            }, *c);
        const auto &vec = std::get<Vec>(data);
        if (vec.size() != 1) throw std::runtime_error("It is impossible to get tuple's typename");
        return std::visit([](const auto& arg) {
            return std::string(typeid(arg).name());
        }, vec[0]);
    }

    inline size_t size() const {
        if (const auto &c = std::get_if<Vec>(&data))
            return std::get<Vec>(data).size();
        return 1;
    }

    // ========= НАСЛЕДИЕ Tensor =========
    // Явно запрещаем таким типам как int, size_t и так далее кастоваться
    // через nullptr->char*->string_view
    template <typename T>
    requires std::convertible_to<T, std::string_view>
    inline Tensor& operator[](const T& name) {
        return ((Tensor*)this)->operator[](name);
    }
    inline const std::vector<size_t> &shape() const { return ((const Tensor*)this)->shape(); }
    inline const std::string &name() const { return ((const Tensor*)this)->name(); }
    inline unsigned int get_id() const { return ((const Tensor*)this)->get_id(); }

    // Функции активации
    Tensor exp() const { return ((const Tensor*)this)->exp(); };
    Tensor relu() const { return ((const Tensor*)this)->relu(); };
    Tensor sigmoid() const { return ((const Tensor*)this)->sigmoid(); };
    Tensor tanh() const { return ((const Tensor*)this)->tanh(); };

    // Возведение в степень
    Tensor pow(float power) const { return ((const Tensor*)this)->pow(power); };

    // Транспонирование
    Tensor transpose() const { return ((const Tensor*)this)->transpose(); };

    inline Tensor &bind() { return ((Tensor*)this)->bind(); }
    template <typename... Args>
    requires (std::convertible_to<Args, int> && ...)
    inline float *touch(Args... other_idxs) const { return ((const Tensor*)this)->touch(other_idxs...); }

    // Создание похожих тензоров
    Tensor like() const { return ((const Tensor*)this)->like(); };

    // Запись состояния в файл
    void repr(const char* filename) const { return ((const Tensor*)this)->repr(filename); }
    void dump(const char* filename) const { return ((const Tensor*)this)->dump(filename); }
    void repr_now(const char* filename) const { return ((const Tensor*)this)->repr_now(filename); }
    void dump_now(const char* filename) const { return ((const Tensor*)this)->dump_now(filename); }

    // Реструктурирование
    Tensor view(const std::vector<size_t> &shape) const { return ((const Tensor*)this)->view(shape); }
    Tensor flatten() const { return ((const Tensor*)this)->flatten(); }

    // Преобразовать тензор в IndexList (только один раз)
    IndexList as_index(unsigned int axis = 0) const { return ((const Tensor*)this)->as_index(axis); }

    // Взятие по индексам
    Tensor gather(const IndexList& indices) const { return ((const Tensor*)this)->gather(indices); }
    Tensor operator[](const IndexList& indices) const { return ((const Tensor*)this)->operator[](indices); }

    // Редуцирующие операции
    Tensor reduce_min(int axis, bool savedim = true) const { return ((const Tensor*)this)->reduce_min(axis, savedim); }
    Tensor reduce_max(int axis, bool savedim = true) const { return ((const Tensor*)this)->reduce_max(axis, savedim); }
    Tensor reduce_sum(int axis, bool savedim = true) const { return ((const Tensor*)this)->reduce_sum(axis, savedim); }
    Tensor reduce_mean(int axis, bool savedim = true) const { return ((const Tensor*)this)->reduce_mean(axis, savedim); }

    Tensor repeat(size_t count) const { return ((const Tensor*)this)->repeat(count); };
};


