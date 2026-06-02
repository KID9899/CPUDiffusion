//
// Created by iliya on 6/3/26.
//

#include <iostream>
#include <random>
#include <cmath>

#include "cpudiff/cpudiff.h"

class ModuleClass(MultiHeadAttention) {
private:
    size_t n_heads;
    std::unique_ptr<ModuleList> q_layers, k_layers, v_layers, o_layers;
public:
    MultiHeadAttention(size_t n_heads): n_heads(n_heads) {
        std::vector<std::vector<std::shared_ptr<Module>>> layers(4);
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < n_heads; ++j) {
                layers[i].push_back(std::make_shared<LinearMatrix>());
            }
        }
        q_layers = std::make_unique<ModuleList>(layers[0]);
        k_layers = std::make_unique<ModuleList>(layers[1]);
        v_layers = std::make_unique<ModuleList>(layers[2]);
        o_layers = std::make_unique<ModuleList>(layers[3]);
        register_module("q_layers", q_layers.get());
        register_module("k_layers", k_layers.get());
        register_module("v_layers", v_layers.get());
        register_module("o_layers", o_layers.get());
    }
    AnyTensor forward(const Tensor &x, const Tensor &casual_mask) {
        size_t head_dim = x.shape().back() / n_heads;
        Tensor out;
        for (size_t i = 0; i < n_heads; ++i) {
            Tensor Q = q_layers->get(i)(x);
            Tensor K = k_layers->get(i)(x);
            Tensor V = v_layers->get(i)(x);

            Tensor scores = (Q ^ K.transpose()) / std::sqrt(static_cast<float>(head_dim));
            scores = scores + casual_mask;
            Tensor attn = CDMath::softmax(scores, -1);
            Tensor head_out = o_layers->get(i)(attn ^ V);
            if (!out) {
                out = head_out;
            } else {
                out = out + head_out;
            }
        }
        return out;
    }
};

int p = 0;

class ModuleClass(TransformerBlock) {
private:
    MultiHeadAttention attn;
    Sequential ff;
    LayerNorm norm1, norm2;
public:
    TransformerBlock(size_t n_heads):
            attn(n_heads),
            ff({
                       std::make_shared<Linear>(),
                       std::make_shared<ReLU>(),
                       std::make_shared<Linear>()
               }),
            norm1(), norm2()
    {
        register_module("attn", attn);
        register_module("ff", ff);
        register_module("norm1", norm1);
        register_module("norm2", norm2);
    }
    AnyTensor forward(Tensor x, const Tensor &casual_mask) {
        int pp = p++;
        x = x + attn(norm1(x), casual_mask);
        x = x + ff(norm2(x));
        return {x, casual_mask};
    }
};

class ModuleClass(SimpleTransformer) {
private:
    Embedding token_emb, pos_emb;
    std::unique_ptr<Sequential> blocks;
    LayerNorm norm;
    Linear out_proj;
public:
    SimpleTransformer(size_t n_heads, size_t n_layers) {
        std::vector<std::shared_ptr<Module>> layers;
        for (size_t i = 0; i < n_layers; ++i) {
            layers.push_back(std::make_shared<TransformerBlock>(n_heads));
        }
        blocks = std::make_unique<Sequential>(layers);
        register_module("token_emb", token_emb);
        register_module("pos_emb", pos_emb);
        register_module("blocks", blocks.get());
        register_module("norm", norm);
        register_module("out_proj", out_proj);
    }
    Tensor forward(const Tensor &x, const Tensor &pos, const Tensor &casual_mask) {
        Tensor emb = token_emb(x) + pos_emb(pos);
        Tensor res = (blocks->call(emb, casual_mask))[0];
        emb = norm(res);
        return out_proj(emb);
    }
};

int main() {

    auto generate_text = [](const std::string &prompt) {
        Graph g;

        SafeTensorsFile st_file("simple_transformer.safetensors");
        Tokenizer tokenizer("simple_transformer_vocab.txt");
        tokenizer.load_merges("simple_transformer_merges.txt");

        SimpleTransformer transformer(4, 6);
        transformer.load(&st_file);

        Tensor tokens = tokenizer(prompt)["input"];
        tokens.as_index();
        tokens.repr_now("tokens.trepr");

        size_t seq_len = tokens.shape()[0];

        std::vector<float> pos_f(seq_len);
        std::vector<float> casual_mask_f(seq_len * seq_len, 0.f);
        for (size_t i = 0; i < seq_len; ++i) {
            pos_f[i] = static_cast<float>(i);
            for (size_t j = i + 1; j < seq_len; ++j) {
                casual_mask_f[i * seq_len + j] = -std::numeric_limits<float>::infinity();
            }
        }


        Tensor pos = g.link(pos_f.data(), {seq_len}, "position range");
        Tensor casual_mask = g.link(casual_mask_f.data(), {seq_len, seq_len}, "casual mask");


        Tensor logits = transformer(tokens, pos, casual_mask)["logits"];
        Tensor probs = CDMath::softmax(logits, -1)["probs"].bind();

        g.compile();
        g.repr("graph.dot");
        g.repr_compiled("compiled.grepr");
        SimpleExecutor se(&g);
        se.execute();

        const float *res = probs.touch(-1);
        size_t vocab = probs.shape().back();

        std::random_device rd;
        std::mt19937 gen(rd());

        std::discrete_distribution<int> dist(res, res + vocab);

        int idx = dist(gen);
        return tokenizer.decode(idx);
    };

    std::string prompt;
    std::getline(std::cin, prompt);

    std::cout << prompt << std::flush;

    for (size_t i = 0; i < 100; ++i) {
        std::string next = generate_text(prompt);
        if (next == "</s>") break;
        std::cout << next << std::flush;
        prompt += next;
    }
    std::cout << std::endl;

    return 0;
}
