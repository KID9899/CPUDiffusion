//
// Created by iliya on 5/20/26.
//

#include <iostream>
#include "cpudiff/cpudiff.h"

constexpr size_t latent_dim = 8;

class ModuleClass(Vae) {
public:
    Sequential encoder, decoder;
    Linear mu, logvar;

    Vae():
            encoder({
                            std::make_shared<Flatten>(),
                            std::make_shared<Linear>(),
                            std::make_shared<ReLU>(),
                            std::make_shared<Linear>(),
                            std::make_shared<ReLU>()
                    }),
            decoder({
                            std::make_shared<Linear>(),
                            std::make_shared<ReLU>(),
                            std::make_shared<Linear>(),
                            std::make_shared<ReLU>(),
                            std::make_shared<Linear>(),
                            std::make_shared<Sigmoid>()
                    })
    {
        register_module("encoder", encoder);
        register_module("mu", mu);
        register_module("logvar", logvar);
        register_module("decoder", decoder);
    }

    Tensor forward(const Tensor &x) {
        Tensor h = encoder(x);
        Tensor s = (logvar(h) / 2.f).exp();
        Tensor r = decoder(Tensor::randn_like(s)*s + mu(h));
        return r.view({0, 28, 28});
    }
};

int main() {
    Graph g;

    SafeTensorsFile vae_file("simple_vae_mnist.safetensors");

    Vae v;
    v.load(&vae_file);

    Tensor x = g.allocate({16, latent_dim}, "X");
    x.randn();

    Tensor y = v.decoder(x).view({0, 28, 28})["Y"];

    y.repr("images.trepr");
    y.bind();

    g.compile();
    g.repr("graph.dot");
    g.repr_compiled("compiled.grepr");
    SimpleExecutor se(&g);
    se.execute();

    CpuDiffImages::save_images(y, "out/%0.png");

    return 0;
}
