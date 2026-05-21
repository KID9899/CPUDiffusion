//
// Created by iliya on 5/20/26.
//

#include <iostream>
#include "cpudiff/cpudiff.h"

static constexpr uint64_t latent_dim = 8;

class Vae final: public Module {
public:
    Tensor s, e, m;
    Sequential encoder, decoder;
    Linear mu, logvar;

    Vae(Graph *graph): Module(graph),
        encoder(graph, {
            std::make_shared<Flatten>(graph),
            std::make_shared<Linear>(graph, 28*28, 256),
            std::make_shared<ReLU>(graph),
            std::make_shared<Linear>(graph, 256, 128),
            std::make_shared<ReLU>(graph)
        }),
        mu(graph, 128, latent_dim),
        logvar(graph, 128, latent_dim),
        decoder(graph, {
            std::make_shared<Linear>(graph, latent_dim, 128),
            std::make_shared<ReLU>(graph),
            std::make_shared<Linear>(graph, 128, 256),
            std::make_shared<ReLU>(graph),
            std::make_shared<Linear>(graph, 256, 28*28),
            std::make_shared<Sigmoid>(graph)
        })
    {
        s = future();
        m = future();
        e = future();
        register_module("encoder", encoder);
        register_module("mu", mu);
        register_module("logvar", logvar);
        register_module("decoder", decoder);
    }

    TensorResult forward(const Tensor &x) override {
        TensorResult h = encoder(x);
        s = logvar(h)[s] / 2.f;
        s = s.exp();
        e.randn();
        e = e * s;
        return e + mu(h)[m];
    }
};

int main() {
    Graph g;
    SafeTensorsFile vae_file("simple_vae_mnist.safetensors", &g);

    Vae v(&g);
    v.load(&vae_file);

    Tensor x = g.allocate({16, latent_dim});
    Tensor ans = g.future();

    x.randn();
    ans = v.decoder.build(x);

    g.dump_dot("test.dot", &vae_file);
    return 0;
}
