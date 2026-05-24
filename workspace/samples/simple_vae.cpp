//
// Created by iliya on 5/20/26.
//

#include <iostream>
#include <fstream>
#include "cpudiff/cpudiff.h"

static constexpr size_t latent_dim = 8;

class Vae final: public Module {
public:
    Tensor s, e, m, h, r;
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
        e = future();
        m = future();
        h = future();
        r = future();
        register_module("encoder", encoder);
        register_module("mu", mu);
        register_module("logvar", logvar);
        register_module("decoder", decoder);
    }

    Tensor::CanAssign forward(const Tensor &x) override {
        h = encoder(x);
        s = (s = logvar(h)) / 2.f;
        s = s.exp();
        e << s.shape();
        e.randn();
        e = e * s;
        e = e + (m = mu(h));
        r = decoder(e);
        return r.view({0, 28, 28});
    }
};

int main() {
    Graph g;
    SafeTensorsFile vae_file("simple_vae_mnist.safetensors", &g);

    Vae v(&g);
    v.load(&vae_file);

    Tensor x = g.allocate({16, latent_dim}, "X");
    Tensor y = g.future("Y");

    x.randn();
    y = v.decoder(x);

    Tensor tmp = g.future("images");
    tmp = y.view({0, 28, 280});
    tmp.repr("images.trepr");

    g.dump_dot("test.dot");
    SimpleExecutor se(&g);
    se.execute();

    CpuDiffImages::save_images(tmp, "out/%0.png");

    return 0;
}
