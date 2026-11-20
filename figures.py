# Figures of the report (run after evaluate.py). Saved in results/figures/.
import csv
from toys import *

COLORS = {"relu": "#2a78d6", "tanh": "#eb6834", "sigmoid": "#1baf7a", "silu": "#eda100", "sin": "#e87ba4"}
SDF_CMAP = mpl.colors.LinearSegmentedColormap.from_list("SDF", [(0, "#eff3ff"), (0.5, "#3182bd"), (0.5, "#31a354"), (1, "#e5f5e0")], N=256)
os.makedirs("results/figures", exist_ok=True)

with open("results/metrics.csv") as file:
    metrics = list(csv.DictReader(file))
with open("results/best_alpha.csv") as file:
    best = list(csv.DictReader(file))


def plot_shapes(networks, toy, ts, filename, figsize=(12, 8.5)):
    """One row per network (label, f), one column per t. Black: learned shape; red dashed: ground truth."""
    res = 64
    x = jgp.sample_2D_grid(res)
    fig, axs = plt.subplots(len(networks), len(ts), figsize=figsize)
    for i, (label, f) in enumerate(networks):
        for j, t in enumerate(ts):
            ax = axs[i, j]
            sdf = onp.array(f(x, t)).reshape(res, res)
            ax.contourf(sdf, levels=onp.linspace(-0.5, 0.5, 21), cmap=SDF_CMAP, extend="both")
            if sdf.min() < 0 < sdf.max():
                ax.contour(sdf, levels=[0], colors="k")
            if toy != "star_circle":
                true = onp.array(true_sdf(toy, x, t)).reshape(res, res)
                ax.contour(true, levels=[0], colors="r", linestyles="--")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            axs[0, j].set_title(f"t = {t}")
        axs[i, 0].set_ylabel(label, fontsize=9)
    plt.tight_layout()
    plt.savefig("results/figures/" + filename, dpi=200)
    plt.close(fig)


# 1) toy of the paper, with the alpha of the paper
plot_shapes([(name, load("star_circle", name, 1e-6)[0]) for name in ACTS],
            "star_circle", [-1, -0.5, 0, 0.5, 1, 1.5, 2], "star_circle_shapes.png")

# 2) sigmoid on the growing circle for every alpha, with its true Lipschitz bound (1/4)^5 * prod ||W_i||_inf
networks = []
for alpha in ALPHAS:
    f, params = load("circle_grow", "sigmoid", alpha)
    bound = onp.prod([float(np.max(np.sum(np.abs(W), axis=1))) for W, b in params]) / 4**5
    print(f"sigmoid, alpha = {alpha:g}: bound = {bound:.3g}")
    networks.append((("standard" if alpha == 0 else rf"$\alpha$ = {alpha:g}") + f"\nbound = {bound:.2g}", f))
plot_shapes(networks, "circle_grow", [-1, 0, 0.5, 1, 2], "sigmoid_alpha.png", figsize=(7, 7.4))

# 3) effect of alpha: training error on the toy of the paper, extrapolation error on the two other toys
panels = [("star_circle", "mse_t0", "MSE at t = 0 (training shape)"),
          ("circle_grow", "mse_extrap", "MSE in extrapolation"),
          ("star_grow", "mse_extrap", "MSE in extrapolation")]
fig, axs = plt.subplots(1, len(panels), figsize=(12, 3.8))
for ax, (toy, column, ylabel) in zip(axs, panels):
    for name in ACTS:
        values = [float(row[column]) for row in metrics if row["toy"] == toy and row["activation"] == name]  # in the order of ALPHAS
        ax.semilogy(range(len(ALPHAS)), values, "o-", color=COLORS[name], label=name)
    ax.set_xticks(range(len(ALPHAS)), ["0\n(standard)", "1e-7", "1e-6", "1e-5", "1e-4"])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_title(toy)
    ax.spines[["top", "right"]].set_visible(False)
axs[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.savefig("results/figures/alpha_summary.png", dpi=200)
plt.close(fig)

# 4) growing circle and growing star, each activation at its selected alpha
for toy in ["circle_grow", "star_grow"]:
    networks = []
    for row in best:
        if row["toy"] == toy:
            alpha = float(row["alpha"])
            networks.append((f"{row['activation']}\nalpha = {alpha:g}", load(toy, row["activation"], alpha)[0]))
    plot_shapes(networks, toy, [-1, -0.5, 0, 0.5, 1, 1.5, 2], f"{toy}_best_shapes.png")

print("saved: results/figures/")
