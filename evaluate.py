# MSE of every trained network, and choice of alpha for each activation (run after main_lipmlp.py).
# Writes results/metrics.csv and results/best_alpha.csv.
import csv
from toys import *

x = jgp.sample_2D_grid(128)                     # 128 x 128 grid on [0,1]^2
ts = onp.round(onp.linspace(-1, 2, 61), 2)      # t from -1 to 2, step 0.05
interp = (ts >= 0) & (ts <= 1)                  # interpolation; the rest is extrapolation

# ---- MSE to the true SDF ----
metrics = []
for toy in TOYS:
    for name in ACTS:
        for alpha in ALPHAS:
            f, _ = load(toy, name, alpha)
            row = {"toy": toy, "activation": name, "alpha": alpha}
            if toy == "star_circle":            # ground truth only at t = 0 and t = 1
                row["mse_t0"] = float(np.mean((f(x, 0) - true_sdf(toy, x, 0)) ** 2))
                row["mse_t1"] = float(np.mean((f(x, 1) - true_sdf(toy, x, 1)) ** 2))
            else:
                mse = onp.array([float(np.mean((f(x, t) - true_sdf(toy, x, t)) ** 2)) for t in ts])
                row["mse_t0"], row["mse_t1"] = mse[ts == 0][0], mse[ts == 1][0]
                row["mse_interp"], row["mse_extrap"] = mse[interp].mean(), mse[~interp].mean()
            metrics.append(row)
            print(row)

# ---- choice of alpha ----
# the largest alpha > 0 whose error at the training codes stays below 2 x that of the standard MLP;
# if there is none, the alpha with the smallest error at the training codes (criterion_met = 0)
def fit(row):
    return (row["mse_t0"] + row["mse_t1"]) / 2

best = []
for toy in TOYS:
    for name in ACTS:
        runs = {row["alpha"]: row for row in metrics if row["toy"] == toy and row["activation"] == name}
        ok = [alpha for alpha in ALPHAS[1:] if fit(runs[alpha]) <= 2 * fit(runs[0.0])]
        alpha = max(ok) if ok else min(ALPHAS[1:], key=lambda a: fit(runs[a]))
        best.append({**runs[alpha], "criterion_met": int(len(ok) > 0)})

columns = ["toy", "activation", "alpha", "mse_t0", "mse_t1", "mse_interp", "mse_extrap"]
for filename, rows, cols in [("metrics.csv", metrics, columns), ("best_alpha.csv", best, columns + ["criterion_met"])]:
    with open("results/" + filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print("saved: results/" + filename)
