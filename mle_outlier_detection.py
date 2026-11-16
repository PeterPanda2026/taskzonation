r"""
Outlier detection for su based on the MLE random field (rf_mle.py).

Model
    ln su = f(z)' beta + eps,  eps a Gaussian random field with covariance
    C = sigma^2 [(1-g) R(theta_h, theta_v) + g I]   (separable exp/sqexp R).
    All parameters are MLEs from the 46 training boreholes (Cluster-BID/4).

Because samples are correlated, a sample is judged against what its
neighbours predict, not against the trend alone.

1. Sample outliers - leave-one-out (LOO) kriging residuals
   With K = C / sigma^2 and Q = K^-1 - K^-1 F (F' K^-1 F)^-1 F' K^-1, the
   LOO prediction error of sample i (universal kriging from all other samples,
   trend re-estimated) has the closed form (Dubrule 1983)
       e_i = (Q y)_i / Q_ii,          var(e_i) = sigma^2 / Q_ii
   so  z_i = (Q y)_i / sqrt(sigma^2 Q_ii)  ~ N(0, 1) under the model.
   Flag |z_i| > Bonferroni threshold  Phi^-1(1 - alpha / (2 n)).

2. Borehole outliers - leave-one-borehole-out (LOBO)
   For borehole B with samples b:  e_B = Q_bb^-1 (Q y)_b,  cov = sigma^2 Q_bb^-1,
       d2_B = (Q y)_b' Q_bb^-1 (Q y)_b / sigma^2  ~ chi2(n_B)
   Flag boreholes with p-value < alpha / (number of boreholes).

3. Observed target su (one per target borehole B1-B5)
   Each is predicted from the training data by universal kriging (new measured
   value, nugget included) and flagged if |z| > Phi^-1(1 - alpha / (2 * 5)).

4. Sensitivity - the random field is refitted without the flagged samples and
   the parameter estimates are compared (--no-refit to skip).

Notes
  * The field parameters are held at their full-data MLEs in steps 1-3.
  * Co-located repeat samples switch the nugget on automatically (as rf_mle.py).
  * The hidden su values of the target boreholes are never read.

Outputs
    results/mle_outlier_samples.csv     z-score, LOO prediction and flag per sample
    results/mle_outlier_boreholes.csv   d2, p-value and flag per borehole
    results/mle_outlier_targets.csv     check of the 5 observed target su
    results/mle_outlier_params.csv      MLEs with and without flagged samples
    figures/mle_outlier_profile.png     su vs depth with flagged samples
    figures/mle_outlier_qq.png          normal Q-Q plot of the LOO z-scores
    figures/mle_outlier_boreholes.png   plan view of the borehole test

Usage
    python mle_outlier_detection.py                 # degree-3 trend (BIC choice in rf_mle.py)
    python mle_outlier_detection.py --alpha 0.01
    python mle_outlier_detection.py --degree 2 --corr sqexp --no-refit
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2, norm

from plot_prediction_data import TARGETS, load_targets
from rf_mle import RandomFieldMLE, correlation, load_data, trend_basis

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
RES_DIR = HERE / "results"
PREFIX = "mle_outlier"


def fit_field(train, args):
    coords = train[["x", "y", "_z"]].to_numpy(float)
    y = train["_y"].to_numpy()
    probe = RandomFieldMLE(coords, y, 0)
    nugget = args.nugget or probe.n_colocated > 0
    mdl = RandomFieldMLE(coords, y, args.degree, "z", args.corr, "estimate", nugget, "ml")
    return mdl, mdl.fit("su")


def describe(f):
    return (f"sigma_ln={math.sqrt(f.sigma2):.3f}, theta_h={f.theta_h:.3g} m, theta_v={f.theta_v:.3g} m, "
            f"nugget={f.nugget:.3f}")


def kriging_diagnostics(mdl, f):
    """Q matrix pieces for LOO / LOBO and the GLS quantities for prediction."""
    K = mdl.K(f.theta_h, f.theta_v, f.nugget)
    cf = cho_factor(K, lower=True)
    Ki = cho_solve(cf, np.eye(mdl.n))
    KiF = Ki @ mdl.F
    A = mdl.F.T @ KiF
    Q = Ki - KiF @ np.linalg.solve(A, KiF.T)
    return Q, Q @ mdl.y, cf, A


def predict_targets(mdl, f, cf, A, obs):
    """Universal-kriging prediction of a new measured ln su at the observed target depths."""
    xu = obs[["x", "y", "_z"]].to_numpy(float)
    xo = mdl.coords
    dh = np.hypot(xu[:, None, 0] - xo[None, :, 0], xu[:, None, 1] - xo[None, :, 1])
    dv = np.abs(xu[:, None, 2] - xo[None, :, 2])
    k = (1 - f.nugget) * correlation(dh, dv, f.theta_h, f.theta_v, mdl.corr)        # n_u x n
    Fu, _ = trend_basis(xu, f.degree, "z", mdl.ranges)
    Kik = cho_solve(cf, k.T)                                                          # n x n_u
    resid = mdl.y - mdl.F @ f.beta
    mu = Fu @ f.beta + Kik.T @ resid
    u = Fu - (mdl.F.T @ Kik).T
    var = f.sigma2 * (1 - np.einsum("ij,ji->i", k, Kik) + np.einsum("ij,jk,ik->i", u, np.linalg.inv(A), u))
    return mu, np.maximum(var, 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=HERE / "BM1_Cluster_BID_4_minus_5BHs.xlsx")
    ap.add_argument("--vertical", default="z_depth", choices=["z_depth", "elevation"])
    ap.add_argument("--degree", type=int, default=3, help="polynomial trend degree in depth")
    ap.add_argument("--corr", default="exp", choices=["exp", "sqexp"])
    ap.add_argument("--nugget", action="store_true", help="force a nugget (auto-on if samples are co-located)")
    ap.add_argument("--alpha", type=float, default=0.05, help="family-wise significance level (Bonferroni)")
    ap.add_argument("--no-refit", action="store_true", help="skip the refit without flagged samples")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):      # absent in some IDE consoles
        sys.stdout.reconfigure(line_buffering=True)
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)

    train = load_data(args.file)
    train = train[train["su"] > 0].reset_index(drop=True)
    train["_z"] = train[args.vertical].astype(float)
    train["_y"] = np.log(train["su"])
    n, nb = len(train), train["borehole_id"].nunique()
    print(f"Training: {nb} boreholes, {n} su values ({args.file.name})")

    print(f"Fitting the random field (degree-{args.degree} trend, {args.corr}) ...")
    mdl, f = fit_field(train, args)
    print(f"  {describe(f)}")
    Q, Qy, cf, A = kriging_diagnostics(mdl, f)

    # 1. sample outliers
    q = np.diag(Q)
    z = Qy / np.sqrt(f.sigma2 * q)
    zcrit = norm.isf(args.alpha / (2 * n))
    samples = train[["global_id", "borehole_id", "x", "y", "z_depth", "elevation", "su"]].copy()
    samples["loo_pred_su"] = np.exp(train["_y"] - Qy / q)
    samples["z_loo"] = z
    samples["p_two_sided"] = 2 * norm.sf(np.abs(z))
    samples["outlier"] = np.abs(z) > zcrit
    samples.to_csv(RES_DIR / f"{PREFIX}_samples.csv", index=False)
    flagged = samples[samples["outlier"]].sort_values("z_loo", key=np.abs, ascending=False)
    print(f"\n1. Sample outliers: |z| > {zcrit:.2f} (alpha={args.alpha}, Bonferroni over {n}): {len(flagged)}")
    if len(flagged):
        print(flagged[["borehole_id", "z_depth", "su", "loo_pred_su", "z_loo"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"   (|z| > 3: {(np.abs(z) > 3).sum()} samples; expected under the model: {2 * norm.sf(3) * n:.1f})")

    # 2. borehole outliers
    rows = []
    for bh, idx in train.groupby("borehole_id").indices.items():
        Qbb = Q[np.ix_(idx, idx)]
        d2 = float(Qy[idx] @ np.linalg.solve(Qbb, Qy[idx]) / f.sigma2)
        e = np.linalg.solve(Qbb, Qy[idx])
        rows.append({"borehole_id": bh, "x": train["x"].iloc[idx[0]], "y": train["y"].iloc[idx[0]],
                     "n": len(idx), "mean_lobo_error_ln": float(e.mean()), "d2": d2,
                     "p_value": float(chi2.sf(d2, len(idx)))})
    bhs = pd.DataFrame(rows)
    bhs["outlier"] = bhs["p_value"] < args.alpha / nb
    bhs.to_csv(RES_DIR / f"{PREFIX}_boreholes.csv", index=False)
    print(f"\n2. Borehole outliers: p < {args.alpha / nb:.2g} (Bonferroni over {nb}): {bhs['outlier'].sum()}")
    print(bhs.nsmallest(5, "p_value")[["borehole_id", "n", "mean_lobo_error_ln", "d2", "p_value", "outlier"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # 3. observed target su
    targets = load_targets()
    obs = targets[targets["su_observed"]].copy()
    obs["_z"] = obs[args.vertical].astype(float)
    mu, var = predict_targets(mdl, f, cf, A, obs)
    obs["pred_su_median"] = np.exp(mu)
    obs["z_score"] = (np.log(obs["su"]) - mu) / np.sqrt(var)
    tcrit = norm.isf(args.alpha / (2 * len(obs)))
    obs["outlier"] = np.abs(obs["z_score"]) > tcrit
    tgt = obs[["target", "borehole_id", "z_depth", "su", "pred_su_median", "z_score", "outlier"]]
    tgt.to_csv(RES_DIR / f"{PREFIX}_targets.csv", index=False)
    print(f"\n3. Observed target su vs. kriging from training (|z| > {tcrit:.2f} flagged):")
    print(tgt.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # 4. sensitivity
    par = [{"fit": "all samples", "n": n, "sigma_ln": math.sqrt(f.sigma2), "theta_h_m": f.theta_h,
            "theta_v_m": f.theta_v, "nugget": f.nugget, **{f"beta_{l}": b for l, b in zip(f.labels, f.beta)}}]
    if len(flagged) and not args.no_refit:
        print(f"\n4. Refitting without the {len(flagged)} flagged sample(s) ...")
        _, f2 = fit_field(train[~samples["outlier"]].reset_index(drop=True), args)
        print(f"   all samples : {describe(f)}")
        print(f"   without     : {describe(f2)}")
        par.append({"fit": "without flagged samples", "n": f2.n, "sigma_ln": math.sqrt(f2.sigma2),
                    "theta_h_m": f2.theta_h, "theta_v_m": f2.theta_v, "nugget": f2.nugget,
                    **{f"beta_{l}": b for l, b in zip(f2.labels, f2.beta)}})
    pd.DataFrame(par).to_csv(RES_DIR / f"{PREFIX}_params.csv", index=False)

    # figures
    zlab = "Depth (m)" if args.vertical == "z_depth" else "Elevation (m)"
    zg = np.linspace(train["_z"].min(), train["_z"].max(), 200)
    Fg, _ = trend_basis(np.column_stack([np.zeros_like(zg), np.zeros_like(zg), zg]), f.degree, "z", mdl.ranges)
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    sc = ax.scatter(train["su"], train["_z"], c=np.abs(z), s=10, cmap="viridis", vmax=max(zcrit, 3))
    fig.colorbar(sc, ax=ax, label="|LOO z-score|")
    ax.plot(np.exp(Fg @ f.beta), zg, "k-", lw=1.5, label="MLE trend (median)")
    ax.scatter(flagged["su"], flagged["z_depth" if args.vertical == "z_depth" else "elevation"], s=90,
               facecolor="none", edgecolor="red", lw=1.5, label=f"sample outlier ({len(flagged)})")
    ax.scatter(obs["su"], obs["_z"], marker="D", s=70, c="k", label="observed target su")
    for _, r in obs.iterrows():
        ax.annotate(f"{r['target']} z={r['z_score']:.1f}", (r["su"], r["_z"]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("$s_u$ (kPa)")
    ax.set_ylabel(zlab)
    if args.vertical == "z_depth":
        ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"MLE random field: leave-one-out outliers\n{describe(f)}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{PREFIX}_profile.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    zs = np.sort(z)
    theo = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax.plot(theo, zs, ".", ms=4)
    lim = max(abs(zs).max(), abs(theo).max()) + 0.3
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8)
    for s in (-zcrit, zcrit):
        ax.axhline(s, color="red", ls="--", lw=0.8)
    ax.set_xlabel("standard normal quantile")
    ax.set_ylabel("LOO z-score")
    ax.set_title(f"Q-Q plot (red: Bonferroni |z| = {zcrit:.2f})", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{PREFIX}_qq.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(bhs["x"], bhs["y"], c=-np.log10(bhs["p_value"]), s=60, cmap="magma_r", edgecolor="k")
    fig.colorbar(sc, ax=ax, label="-log10 p (borehole LOBO test)")
    for _, r in bhs[bhs["outlier"]].iterrows():
        ax.annotate(str(r["borehole_id"]), (r["x"], r["y"]), xytext=(5, 5), textcoords="offset points",
                    color="red", weight="bold")
    tb = targets.groupby("borehole_id").first()
    ax.scatter(tb["x"], tb["y"], marker="*", s=200, c="w", edgecolor="k", label="target boreholes")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8)
    ax.set_title(f"Borehole outliers (red labels: p < {args.alpha / nb:.2g})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{PREFIX}_boreholes.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved results/{PREFIX}_*.csv and figures/{PREFIX}_*.png")


if __name__ == "__main__":
    main()
