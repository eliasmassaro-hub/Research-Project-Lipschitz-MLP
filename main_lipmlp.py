# Trains every network of the experiment (runs already trained are skipped).
# Same network, loss and optimiser as main_lipmlp.py of the official demo.
from toys import *


def train(toy, name, alpha, seed=0, num_epochs=200000):
    path = run_path(toy, name, alpha, seed)
    if os.path.exists(path + ".pkl"):
        return
    print(f"\n=== {path} ===")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    random.seed(seed)
    act, init = ACTS[name]
    model = lipmlp({"dim_in": 2, "dim_t": 1, "dim_out": 1, "h_mlp": [64, 64, 64, 64, 64],
                    "act": act, "init": init, "lipschitz": alpha > 0})
    params = model.initialize_weights()
    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-4)
    opt_state = opt_init(params)

    # the network only sees the two training shapes, at t = 0 and t = 1
    def loss(params, alpha, x, y0, y1):
        mse = np.mean((model.forward(params, np.array([0.0]), x) - y0) ** 2) \
            + np.mean((model.forward(params, np.array([1.0]), x) - y1) ** 2)
        return mse + alpha * model.get_lipschitz_loss(params)

    @jit
    def update(epoch, opt_state, alpha, x, y0, y1):
        value, grads = value_and_grad(loss)(get_params(opt_state), alpha, x, y0, y1)
        return value, opt_update(epoch, grads, opt_state)

    for epoch in tqdm.tqdm(range(num_epochs)):
        x = np.array(random.rand(512, 2))          # 512 random points in [0,1]^2
        value, opt_state = update(epoch, opt_state, alpha, x, true_sdf(toy, x, 0), true_sdf(toy, x, 1))
    print(f"final loss: {value:.2e}")

    with open(path + ".pkl", "wb") as file:
        pickle.dump(get_params(opt_state), file)


if __name__ == "__main__":
    for toy in TOYS:
        for alpha in ALPHAS:
            for name in ACTS:
                train(toy, name, alpha)
