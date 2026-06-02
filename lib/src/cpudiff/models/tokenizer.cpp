//
// Created by iliya on 5/31/26.
//

#include "tokenizer.h"

void Tokenizer::init_mappings() {
    std::vector<uint8_t> bs;
    std::vector<char32_t> cs;

    // печатные ASCII: от '!' (33) до '~' (126)
    for (int i = 33; i <= 126; ++i) bs.push_back(i);
    // Latin-1: от '¡' (161) до '¬' (172)
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    // Latin-1: от '®' (174) до 'ÿ' (255)
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    cs.assign(bs.begin(), bs.end());

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    for (size_t i = 0; i < 256; ++i) {
        byte_to_unicode_[bs[i]] = cs[i];
        unicode_to_byte_[cs[i]] = bs[i];
    }
}

std::u32string Tokenizer::utf8_to_u32(const std::string &utf8) {
    std::u32string result;
    size_t i = 0;
    while (i < utf8.size()) {
        unsigned char c = utf8[i];
        char32_t codepoint;
        size_t extra;
        if (c < 0x80) {
            codepoint = c;
            extra = 0;
        } else if (c > 0xC0 && c < 0xE0) {
            codepoint = c & 0x1F;
            extra = 1;
        } else if (c > 0xC0 && c < 0xF0) {
            codepoint = c & 0x0F;
            extra = 2;
        } else if (c > 0xC0 && c < 0xF8) {
            codepoint = c & 0x07;
            extra = 3;
        } else {
            throw std::runtime_error("Invalid UTF-8");
        }
        if (i + extra >= utf8.size())
            throw std::runtime_error("Invalid UTF-8");
        for (size_t j = 1; j <= extra; ++j) {
            unsigned char cont = utf8[i + j];
            if ((cont & 0xC0) != 0x80)
                throw std::runtime_error("Invalid UTF-8");
            codepoint = (codepoint << 6) | (cont & 0x3F);
        }
        result.push_back(codepoint);
        i += 1 + extra;
    }
    return result;
}

void Tokenizer::load(const std::string &filename, bool bpe) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    tokens_.clear();
    token_to_id_.clear();
    std::string line;
    int idx = 0;

    while (std::getline(file, line)) {
        // пропускаем пустые строки
        if (line.empty() && file.eof()) break;
        std::string token_bytes;

        if (bpe) {
            std::u32string u32line = utf8_to_u32(line);
            for (char32_t ch : u32line) {
                auto it = unicode_to_byte_.find(ch);
                if (it == unicode_to_byte_.end())
                    throw std::runtime_error("Unmapped character in token");
                token_bytes.push_back(static_cast<char>(it->second));
            }
        } else {
            token_bytes = line; // без преобразования
        }

        tokens_.push_back(token_bytes);
        token_to_id_[token_bytes] = idx++;
    }
}

void Tokenizer::load_merges(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open merges file: " + filename);

    merges_.clear();
    merge_rank_.clear();
    std::string line;
    int rank = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string t1, t2;
        if (!(iss >> t1 >> t2)) continue;   // строка должна содержать два слова

        std::u32string ut1 = utf8_to_u32(t1);
        std::u32string ut2 = utf8_to_u32(t2);
        merges_.emplace_back(ut1, ut2);
        merge_rank_[{ut1, ut2}] = rank++;
    }
}

Tensor Tokenizer::tokenize(const std::string &text, bool use_bpe) const {
    if (use_bpe && merges_.empty())
        throw std::runtime_error("Merges not loaded. Call load_merges() first.");

    // 1. текст -> байты -> Unicode-строка (byte_to_unicode_)
    std::u32string unicode_str;
    for (unsigned char byte : text) {
        unicode_str.push_back(byte_to_unicode_[byte]);
    }

    std::vector<int> result;
    if (!use_bpe) {
        result = greedy_tokenize(unicode_str);
    } else {

        // 2. начальное разбиение на отдельные символы
        std::vector<std::u32string> tokens;
        for (char32_t ch: unicode_str) {
            tokens.emplace_back(1, ch);
        }

        // 3. итеративные слияния BPE
        while (tokens.size() > 1) {
            int best_rank = INT_MAX;
            size_t best_i = 0;
            // ищем пару с наименьшим рангом (наивысший приоритет)
            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                auto it = merge_rank_.find({tokens[i], tokens[i + 1]});
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }
            if (best_rank == INT_MAX) break; // больше нечего сливать

            // выполняем слияние
            tokens[best_i] = tokens[best_i] + tokens[best_i + 1];
            tokens.erase(tokens.begin() + static_cast<ptrdiff_t>(best_i) + 1);
        }

        // 4. преобразуем каждый токен обратно в байты и находим его индекс
        for (const auto &utoken: tokens) {
            std::string byte_token;
            for (char32_t ch: utoken) {
                auto it = unicode_to_byte_.find(ch);
                if (it == unicode_to_byte_.end())
                    throw std::runtime_error("Internal error: unmapped character");
                byte_token.push_back(static_cast<char>(it->second));
            }
            auto id_it = token_to_id_.find(byte_token);
            if (id_it == token_to_id_.end())
                throw std::runtime_error("Unknown token: " + byte_token);
            result.push_back(id_it->second);
        }
    }
    Tensor res_t = graph->allocate(
            {result.size()},
            text.length() > 23
            ? text.substr(0, 10) + "..." + text.substr(text.length() - 10, 10)
            : text
    );
    float *res_a = graph->force_bind(res_t, false);
    for (size_t i = 0; i < result.size(); ++i) {
        res_a[i] = static_cast<float>(result[i]);
    }
    return res_t;
}

std::string Tokenizer::decode(int id, bool raw) const {
    if (id < 0 || id >= static_cast<int>(tokens_.size()))
        throw std::out_of_range("Token id out of range");
    const std::string& byte_token = tokens_[id];
    if (!raw) return byte_token;
    std::u32string unicode_token;
    for (unsigned char b : byte_token) {
        unicode_token.push_back(byte_to_unicode_[b]);
    }
    // UTF-32 -> UTF-8
    std::string result;
    for (char32_t cp : unicode_token) {
        if (cp < 0x80) {
            result.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            result.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            result.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x110000) {
            result.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            throw std::runtime_error("Invalid code point in token");
        }
    }
    return result;
}

std::vector<int> Tokenizer::greedy_tokenize(const std::u32string &unicode_str) const {
    std::vector<int> result;
    size_t pos = 0;
    while (pos < unicode_str.size()) {
        size_t best_len;
        int best_id = -1;
        // ищем максимальную подстроку, которая есть в словаре
        for (size_t len = std::min(unicode_str.size() - pos, max_token_length()); len >= 1; --len) {
            std::u32string sub = unicode_str.substr(pos, len);
            // преобразуем обратно в байты
            std::string byte_sub = u32_to_bytes(sub);
            auto it = token_to_id_.find(byte_sub);
            if (it != token_to_id_.end()) {
                best_len = len;
                best_id = it->second;
                break; // берём первую максимальную
            }
        }
        if (best_id == -1) {
            throw std::runtime_error("Unknown token at position " + std::to_string(pos));
        }
        result.push_back(best_id);
        pos += best_len;
    }
    return result;
}

std::string Tokenizer::u32_to_bytes(const std::u32string &u32) const {
    std::string bytes;
    for (char32_t ch : u32) {
        auto it = unicode_to_byte_.find(ch);
        if (it == unicode_to_byte_.end())
            throw std::runtime_error("Unmapped character in token");
        bytes.push_back(static_cast<char>(it->second));
    }
    return bytes;
}

size_t Tokenizer::max_token_length() const {
    static size_t max_len = 0;
    if (max_len == 0) {
        for (const auto& t : tokens_)
            max_len = std::max(max_len, t.size());
    }
    return max_len;
}
