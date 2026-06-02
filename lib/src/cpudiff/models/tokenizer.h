//
// Created by iliya on 5/31/26.
//

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <climits>

#include "cpudiff/core.h"

#pragma once

class Tokenizer {
private:
    Graph *graph;

    // отображения byte в unicode char
    std::array<char32_t, 256> byte_to_unicode_{};
    std::unordered_map<char32_t, uint8_t> unicode_to_byte_;

    // словарь токенов в виде байтовых строк и map для быстрого поиска
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int> token_to_id_;

    // пары Unicode-токенов, упорядоченные по приоритету
    std::vector<std::pair<std::u32string, std::u32string>> merges_;
    std::map<std::pair<std::u32string, std::u32string>, int> merge_rank_;

    void init_mappings();

    // UTF-8 -> UTF-32
    static std::u32string utf8_to_u32(const std::string& utf8);


    // Жадное разбиение
    std::vector<int> greedy_tokenize(const std::u32string& unicode_str) const;

    std::string u32_to_bytes(const std::u32string& u32) const;

    // максимальная длина токена в байтовом представлении
    size_t max_token_length() const;
public:
    explicit Tokenizer(Graph *graph): graph(graph) { init_mappings(); }
    Tokenizer(): Tokenizer(Graph::get_active()) {};
    Tokenizer(Graph *graph, const std::string& filename, bool bpe = true): Tokenizer(graph) { load(filename, bpe); }
    explicit Tokenizer(const std::string& filename, bool bpe = true): Tokenizer(Graph::get_active(), filename, bpe) {};

    // загрузка словаря токенов
    void load(const std::string& filename, bool bpe = true);

    // загрузка таблицы слияний BPE
    void load_merges(const std::string& filename);

    // токенизация входного текста
    Tensor tokenize(const std::string& text, bool use_bpe = false) const;
    inline Tensor operator()(const std::string& text) const { return tokenize(text); }

    // Получить текстовое представление токена по его индексу
    std::string decode(int id, bool raw = false) const;
};
