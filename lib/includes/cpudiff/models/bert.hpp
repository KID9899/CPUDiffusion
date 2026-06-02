//
// Created by iliya on 5/28/26.
//

#include "cpudiff/cpudiff.h"

class BertEmbeddings : public Module {
    Embedding word_embeddings;
    Embedding position_embeddings;
    Embedding token_type_embeddings;
    LayerNorm layer_norm;
public:
    // Конструктор 1 – создаёт все подмодули с заданными размерами
    BertEmbeddings(Graph* g): Module(g),
                              word_embeddings(g),
                              position_embeddings(g),
                              token_type_embeddings(g),
                              layer_norm(g)
    {
        register_module("word_embeddings", word_embeddings);
        register_module("position_embeddings", position_embeddings);
        register_module("token_type_embeddings", token_type_embeddings);
        register_module("LayerNorm", layer_norm);
    }
    Tensor forward(const Tensor&) override { throw std::logic_error("!"); };
    Tensor forward(const Tensor& input_ids,
                   const Tensor& token_type_ids,
                   const Tensor& position_ids) {
        // Все входы предполагаются двумерными [batch, seq_len]
        size_t batch = input_ids.shape()[0];
        size_t seq_len = input_ids.shape()[1];
        size_t hidden = word_embeddings(input_ids).shape()[2]; // не очень надёжно, но ок

        Tensor w = word_embeddings(input_ids);                     // [batch, seq_len, hidden]
        Tensor p = position_embeddings(position_ids);              // [batch, seq_len, hidden]
        Tensor t = token_type_embeddings(token_type_ids);          // [batch, seq_len, hidden]

        Tensor sum = w + p + t;
        return layer_norm(sum);
    }
protected:
    const char* CLASS_NAME() override { return "BertEmbeddings"; }
};

class BertSelfAttention : public Module {
    Linear query, key, value;
    size_t num_heads;
public:
    BertSelfAttention(Graph* g, size_t num_heads)
            : Module(g),
              query(g),
              key(g),
              value(g),
              num_heads(num_heads) {
        register_module("query", query);
        register_module("key", key);
        register_module("value", value);
    }
    BertSelfAttention(size_t num_heads): BertSelfAttention(Graph::get_active(), num_heads) {}

    Tensor forward(const Tensor& hidden_states) override {
        size_t batch = hidden_states.shape()[0];
        size_t seq_len = hidden_states.shape()[1];

        Tensor q = query(hidden_states).view({batch, seq_len, num_heads, 0});
        Tensor k = key(hidden_states).view({batch, seq_len, num_heads, 0});
        Tensor v = value(hidden_states).view({batch, seq_len, num_heads, 0});

        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);

        Tensor k_t = k.transpose(-2, -1);
        Tensor scores = (q ^ k_t) / q.shape().back();
        Tensor attn_probs = scores.softmax(-1);
        Tensor context = attn_probs ^ v;

        context = context.transpose(1, 2).view({batch, seq_len, 0});
        return context;
    }

    const char* CLASS_NAME() override { return "BertSelfAttention"; }
};

class BertSelfOutput : public Module {
    Linear dense;
    LayerNorm layer_norm;
public:
    BertSelfOutput(Graph* g): Module(g), dense(g), layer_norm(g) {
        register_module("dense", dense);
        register_module("LayerNorm", layer_norm);
    }

    Tensor forward(const Tensor&) override { throw std::logic_error("!"); };
    Tensor forward(const Tensor& hidden_states, const Tensor& input_tensor) {
        return layer_norm(dense(hidden_states) + input_tensor);
    }

    const char* CLASS_NAME() override { return "BertSelfOutput"; }
};

class BertAttention : public Module {
    BertSelfAttention self;
    BertSelfOutput output;
public:
    BertAttention(Graph* g, size_t num_heads): Module(g), self(g, num_heads), output(g) {
        register_module("self", self);
        register_module("output", output);
    }

    Tensor forward(const Tensor& hidden_states) {
        Tensor self_output = self(hidden_states);
        return output.forward(self_output, hidden_states);
    }
    const char* CLASS_NAME() override { return "BertAttention"; }
};

class BertIntermediate : public Module {
    Linear dense;
    GELU gelu;
public:
    BertIntermediate(Graph* g): Module(g), dense(g) {
        register_module("dense", dense);
    }

    Tensor forward(const Tensor& hidden_states) override {
        return gelu(dense(hidden_states));
    }
    const char* CLASS_NAME() override { return "BertIntermediate"; }
};

class BertOutput : public Module {
    Linear dense;
    LayerNorm layer_norm;
public:
    BertOutput(Graph* g): Module(g), dense(g), layer_norm(g) {
        register_module("dense", dense);
        register_module("LayerNorm", layer_norm);
    }

    Tensor forward(const Tensor&) override { throw std::logic_error("!"); };
    Tensor forward(const Tensor& hidden_states, const Tensor& input_tensor) {
        return layer_norm(dense(hidden_states) + input_tensor);
    }
    const char* CLASS_NAME() override { return "BertOutput"; }
};

class BertLayer : public Module {
    BertAttention attention;
    BertIntermediate intermediate;
    BertOutput output;
public:
    BertLayer(Graph* g, size_t num_heads)
            : Module(g),
              attention(g, num_heads),
              intermediate(g),
              output(g)
    {
        register_module("attention", attention);
        register_module("intermediate", intermediate);
        register_module("output", output);
    }

    Tensor forward(const Tensor& hidden_states) override {
        Tensor attn_output = attention(hidden_states);
        Tensor inter = intermediate(attn_output);
        return output.forward(inter, attn_output);
    }
    const char* CLASS_NAME() override { return "BertLayer"; }
};

class BertEncoder : public Module {
    std::vector<std::unique_ptr<BertLayer>> layers;
public:
    BertEncoder(Graph* g, size_t num_layers, size_t num_heads): Module(g) {
        for (size_t i = 0; i < num_layers; ++i) {
            auto layer = std::make_unique<BertLayer>(g, num_heads);
            register_module("layer." + std::to_string(i), *layer);
            layers.push_back(std::move(layer));
        }
    }

    Tensor forward(const Tensor& hidden_states) override {
        Tensor h = hidden_states;
        for (auto& layer : layers) h = (*layer)(h);
        return h;
    }
    const char* CLASS_NAME() override { return "BertEncoder"; }
};

class BertPooler : public Module {
    Linear dense;
public:
    BertPooler(Graph* g) : Module(g), dense(g) {
        register_module("dense", dense);
    }

    Tensor forward(const Tensor& hidden_states) override {
        size_t batch = hidden_states.shape()[0];
        size_t hidden = hidden_states.shape()[2];
        Tensor cls_idx = graph->allocate({batch, 1});
        cls_idx.fill(0.0f);
        IndexList idx = cls_idx.as_index(1);
        Tensor cls_token = hidden_states.gather(idx).view({batch, 0});
        return dense(cls_token).tanh();
    }
    const char* CLASS_NAME() override { return "BertPooler"; }
};

class BertModel : public Module {
    BertEmbeddings embeddings;
    BertEncoder encoder;
    BertPooler pooler;
public:
    BertModel(Graph* g, size_t num_layers, size_t num_heads)
            : Module(g),
              embeddings(g),
              encoder(g, num_layers, num_heads),
              pooler(g)
    {
        register_module("embeddings", embeddings);
        register_module("encoder", encoder);
        register_module("pooler", pooler);
    }

    Tensor forward(const Tensor&) override { throw std::logic_error("!"); };
    Tensor forward(const Tensor& input_ids, const Tensor& token_type_ids) {
        Tensor emb = embeddings.forward(input_ids, token_type_ids);
        Tensor seq_out = encoder(emb);
        return pooler(seq_out);
    }

    Tensor operator()(const Tensor& input_ids, const Tensor& token_type_ids) {
        return forward(input_ids, token_type_ids);
    }
    const char* CLASS_NAME() override { return "BertModel"; }
};
