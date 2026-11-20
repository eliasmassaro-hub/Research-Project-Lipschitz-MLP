# Shared definitions: activations, values of alpha, toys, and loading of a trained network.
import os, pickle
from model import *

# activation: (function, weight initialisation)
ACTS = {
    "relu":    (jax.nn.relu,    "he"),
    "tanh":    (jax.nn.tanh,    "glorot"),
    "sigmoid": (jax.nn.sigmoid, "glorot"),
    "silu":    (jax.nn.silu,    "he"),
    "sin":     (np.sin,         "glorot"),
}
ALPHAS = [0.0, 1e-7, 1e-6, 1e-5, 1e-4]   # alpha = 0: standard MLP, without the Lipschitz normalisation
TOYS = ["star_circle", "circle_grow", "star_grow"]


def true_sdf(toy, x, t):
    """SDF of the toy at code t, on the points x."""
    if toy == "star_circle":   # toy of the paper: star at t = 0, circle at t = 1 (nothing for other t)
        return jgp.sdf_star(x) if t == 0 else jgp.sdf_circle(x)
    if toy == "circle_grow":   # radius 0.2 at t = 0, 0.3 at t = 1, linear in t
        return jgp.sdf_circle(x, r=0.2 + 0.1 * t)
    if toy == "star_grow":     # size 0.12 at t = 0, 0.17 at t = 1, linear in t
        return jgp.sdf_star(x, r=0.12 + 0.05 * t)


def run_path(toy, name, alpha, seed=0):
    """Files of a run, without extension, e.g. results/circle_grow/tanh_lip1_a1e-06_s0"""
    return f"results/{toy}/{name}_lip{int(alpha > 0)}_a{alpha:g}_s{seed}"


def load(toy, name, alpha):
    """Trained network. Returns f(x, t), the learned SDF, and its weights."""
    model = lipmlp({"act": ACTS[name][0], "lipschitz": alpha > 0})
    with open(run_path(toy, name, alpha) + ".pkl", "rb") as file:
        params = model.normalize_params(pickle.load(file))
    f = lambda x, t: model.forward_eval(params, np.array([float(t)]), x)
    return f, params
