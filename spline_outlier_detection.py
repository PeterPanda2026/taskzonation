r"""
Outlier detection for su based on the coordinates-only spline regression
(spline_select.py or spline_penalized.py).

Model
    ln su = s_z(z) + s_x(x) + s_y(y) + e        (x, y, depth only)
  --method select     regression spline; pieces (K_z, K_x, K_y) chosen by
                      leave-one-borehole-out CV, or fixed with --pieces
  --method penalized  P-spline; (lambda_z, lambda_h) chosen by
                      leave-one-borehole-out CV, or fixed with --lambdas
  Both are linear smoothers  y_hat = H y  with hat matrix H and p = trace(H)
  effective parameters (p = number of columns for the regression spline).

1. Sample outliers - externally studentized residuals
       t_i = r_i / (s_(i) sqrt(1 - h_ii)),
       s_(i)^2 = [(n - p) s^2 - r_i^2 / (1 - h_ii)] / (n - p - 1)
   Flag |t_i| > Bonferroni threshold t_{1 - alpha/(2n), n-p-1}.
   Cook's distance  D_i = r_i^2 h_ii / (p s^2 (1 - h_ii)^2)  measures each
   sample's influence on the fit (D_i > 4/n is listed as influential).
   These tests treat residuals as independent; within a borehole they are
   vertically correlated, so step 2 checks whole boreholes as well.

2. Borehole outliers - leave-one-borehole-out (LOBO) residuals
   Each borehole is predicted from a fit without it; its mean prediction
   error is standardized with a variance that allows for the vertical
   residual correlation rho(dz) = (1 - g) exp(-2|dz|/theta_v)
   (estimated as in spline_select.py):
       var(mean e_B) = sigma^2 / n_B^2 * sum_ij R_ij,     R_ii = 1
   Flag |z_B| > Phi^-1(1 - alpha / (2 * number of boreholes)).

3. Observed target su (one per target borehole B1-B5)
   z = (ln su_obs - prediction) / prediction sd, flagged if
   |z| > Phi^-1(1 - alpha / (2 * 5)).

4. Sensitivity - the spline is refitted (same settings) without the flagged
   samples; the change in sigma and in the fitted depth profile is reported.

The hidden su values of the target boreholes are never read.

Outputs
    results/spline_outlier_<method>_samples.csv     t, leverage, Cook's D, flags
    results/spline_outlier_<method>_boreholes.csv   LOBO borehole test
    results/spline_outlier_<method>_targets.csv     check of the 5 observed target su
    figures/spline_outlier_<method>_profile.png     su vs depth with flagged samples
    figures/spline_outlier_<method>_leverage.png    studentized residual vs leverage
    figures/spline_outlier_<method>_boreholes.png   plan view of the borehole test

Usage
    python spline_outlier_detection.py                         # regression spline, CV choice
    python spline_outlier_detection.py --method penalized
    python spline_outlier_detection.py --pieces 8 0 0          # skip the CV search
    python spline_outlier_detection.py --method penalized --lambdas 10 3.16e7 --alpha 0.01
"""
import argparse
import itertools
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from spline_penalized import PSpline
from spline_select import (FIG_DIR, RES_DIR, RegressionSpline, fit_vertical_correlation, load,
                           loo_borehole_cv, penalized_solve)


# ----------------------------------------------------------------------------
# model choice and fitting (same settings as the two spline scripts)
# ----------------------------------------------------------------------------
def build(args, train, bounds):
    """Return (model, X, penalty P or None, hat diagonal, p_eff, label)."""
    y, groups = train["_y"].to_numpy(), train["borehole_id"].to_numpy()
    if args.method == "select":
        if args.pieces:
            kz, kx, ky = args.pieces
        else:
            print("Choosing (K_z, K_x, K_y) by leave-one-borehole-out CV ...")
            best = None
            for kz, kx, ky in itertools.product(range(1, 16), range(5), range(5)):
                m = RegressionSpline({"z": kz, "x": kx, "y": ky}, 3, "quantile", bounds).fit(train)
                cv = loo_borehole_cv(m.design(train), y, groups)[0]
                if best is None or cv < best[0]:
                    best = (cv, kz, kx, ky)
            _, kz, kx, ky = best
        model = RegressionSpline({"z": kz, "x": kx, "y": ky}, 3, "quantile", bounds).fit(train)
        X, P = model.design(train), None
        h = np.einsum("ij,jk,ik->i", X, model.XtX_inv, X)
        return model, X, P, h, float(model.p), f"regression spline K_z={kz}, K_x={kx}, K_y={ky}"

    segs = {"z": 40, "x": 10, "y": 10}
    base = PSpline(segs, 3, 2, bounds).setup(train)
    if args.lambdas:
        lz, lh = args.lambdas
    else:
        print("Choosing (lambda_z, lambda_h) by leave-one-borehole-out CV ...")
        pairs = list(itertools.product(np.logspace(-2, 5, 22), np.logspace(-1, 8, 19)))
        cv = loo_borehole_cv(base.X, base.y, groups, [base.penalty(a, b) for a, b in pairs])
        lz, lh = pairs[int(np.argmin(cv))]
    model = PSpline(segs, 3, 2, bounds).fit(train, lz, lh)
    X, P = model.X, model.penalty(lz, lh)
    h = np.einsum("ij,jk,ik->i", X, model.Ainv, X)
    return model, X, P, h, model.edf, f"P-spline lambda_z={lz:.3g}, lambda_h={lh:.3g}, edf={model.edf:.1f}"


def refit(args, model, train_sub):
    """Same settings, fitted to a subset of the training data."""
    if args.method == "select":
        return RegressionSpline(model.pieces, model.degree, model.knots_how, model.bounds).fit(train_sub)
    return PSpline(model.segments, model.degree, 2, model.bounds).fit(train_sub, *model.lam)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="select", choices=["select", "penalized"])
    ap.add_argument("--vertical", default="z_depth", choices=["z_depth", "elevation"])
    ap.add_argument("--pieces", type=int, nargs=3, metavar=("K_Z", "K_X", "K_Y"),
                    help="fixed numbers of pieces for --method select (skips the CV search)")
    ap.add_argument("--lambdas", type=float, nargs=2, metavar=("LAMBDA_Z", "LAMBDA_H"),
                    help="fixed penalty weights for --method penalized (skips the CV search)")
    ap.add_argument("--alpha", type=float, default=0.05, help="family-wise significance level (Bonferroni)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):      # absent in some IDE consoles
        sys.stdout.reconfigure(line_buffering=True)
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    prefix = f"spline_outlier_{args.method}"

    train, targets, bounds = load(args.vertical)
    y, groups = train["_y"].to_numpy(), train["borehole_id"].to_numpy()
    n, nb = len(train), train["borehole_id"].nunique()
    print(f"Training: {nb} boreholes, {n} su values; explanatory variables: x, y, {args.vertical}")

    model, X, P, h, p, label = build(args, train, bounds)
    r = model.resid
    s2 = float(r @ r) / (n - p)
    print(f"Model: {label}; sigma_ln = {math.sqrt(s2):.3f}")

    # 1. sample outliers
    s2_i = ((n - p) * s2 - r ** 2 / (1 - h)) / (n - p - 1)
    tstud = r / np.sqrt(s2_i * (1 - h))
    cook = r ** 2 * h / (p * s2 * (1 - h) ** 2)
    tcrit = student_t.isf(args.alpha / (2 * n), n - p - 1)
    samples = train[["global_id", "borehole_id", "x", "y", "z_depth", "elevation", "su"]].copy()
    samples["fitted_su"] = np.exp(y - r)
    samples["leverage"] = h
    samples["t_studentized"] = tstud
    samples["cooks_d"] = cook
    samples["outlier"] = np.abs(tstud) > tcrit
    samples["influential"] = cook > 4 / n
    samples.to_csv(RES_DIR / f"{prefix}_samples.csv", index=False)
    flagged = samples[samples["outlier"]].sort_values("t_studentized", key=np.abs, ascending=False)
    print(f"\n1. Sample outliers: |t| > {tcrit:.2f} (alpha={args.alpha}, Bonferroni over {n}): {len(flagged)}")
    if len(flagged):
        print(flagged[["borehole_id", "z_depth", "su", "fitted_su", "t_studentized", "cooks_d"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"   influential (Cook's D > 4/n = {4 / n:.4f}): {samples['influential'].sum()} samples; "
          f"max D = {cook.max():.3f}")

    # 2. borehole outliers (LOBO, correlation-aware variance)
    corr, _ = fit_vertical_correlation(train, r)
    XtX, Xty = X.T @ X, X.T @ y
    rows = []
    for bh, idx in pd.Series(range(n)).groupby(groups).indices.items():
        Xi, yi = X[idx], y[idx]
        coef = penalized_solve(XtX - Xi.T @ Xi, Xty - Xi.T @ yi, P)
        e = yi - Xi @ coef
        dz = np.abs(train["z"].to_numpy()[idx][:, None] - train["z"].to_numpy()[idx][None, :])
        R = (1 - corr["nugget"]) * np.exp(-2 * dz / corr["theta_v"])
        np.fill_diagonal(R, 1.0)
        sd_mean = math.sqrt(corr["sigma2"] * R.sum()) / len(idx)
        rows.append({"borehole_id": bh, "x": train["x"].iloc[idx[0]], "y": train["y"].iloc[idx[0]],
                     "n": len(idx), "mean_lobo_error_ln": float(e.mean()),
                     "rms_lobo_error_ln": float(np.sqrt(np.mean(e ** 2))), "z_mean": float(e.mean() / sd_mean)})
    bhs = pd.DataFrame(rows)
    bcrit = norm.isf(args.alpha / (2 * nb))
    bhs["p_value"] = 2 * norm.sf(np.abs(bhs["z_mean"]))
    bhs["outlier"] = np.abs(bhs["z_mean"]) > bcrit
    bhs.to_csv(RES_DIR / f"{prefix}_boreholes.csv", index=False)
    print(f"\n2. Borehole outliers: |z| > {bcrit:.2f} (Bonferroni over {nb}; residual correlation "
          f"theta_v={corr['theta_v']:.1f} m, nugget={corr['nugget']:.2f}): {bhs['outlier'].sum()}")
    print(bhs.reindex(bhs["z_mean"].abs().sort_values(ascending=False).index).head(5)
          [["borehole_id", "n", "mean_lobo_error_ln", "rms_lobo_error_ln", "z_mean", "outlier"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # 3. observed target su
    obs = targets[targets["su_observed"]].copy()
    mu, var = model.predict(obs)
    obs["pred_su_median"] = np.exp(mu)
    obs["z_score"] = (obs["_y"] - mu) / np.sqrt(var)
    ocrit = norm.isf(args.alpha / (2 * len(obs)))
    obs["outlier"] = np.abs(obs["z_score"]) > ocrit
    tgt = obs[["target", "borehole_id", "z_depth", "su", "pred_su_median", "z_score", "outlier"]]
    tgt.to_csv(RES_DIR / f"{prefix}_targets.csv", index=False)
    print(f"\n3. Observed target su vs. spline prediction (|z| > {ocrit:.2f} flagged):")
    print(tgt.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # 4. sensitivity
    zg = np.linspace(train["z"].min(), train["z"].max(), 200)
    centre = pd.DataFrame({"z": zg, "x": train["x"].median(), "y": train["y"].median()})
    m_all = model.predict(centre)[0]
    if len(flagged):
        m2 = refit(args, model, train[~samples["outlier"].to_numpy()].reset_index(drop=True))
        r2 = m2.resid
        p2 = m2.p if args.method == "select" else m2.edf
        print(f"\n4. Refit without the {len(flagged)} flagged sample(s): sigma_ln {math.sqrt(s2):.3f} -> "
              f"{math.sqrt(float(r2 @ r2) / (len(r2) - p2)):.3f}; max change of the fitted ln su profile "
              f"at the site centre = {np.max(np.abs(m2.predict(centre)[0] - m_all)):.3f}")

    # figures
    zlab = "Depth (m)" if args.vertical == "z_depth" else "Elevation (m)"
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    sc = ax.scatter(train["su"], train["z"], c=np.abs(tstud), s=10, cmap="viridis", vmax=max(tcrit, 3))
    fig.colorbar(sc, ax=ax, label="|studentized residual|")
    ax.plot(np.exp(m_all), zg, "k-", lw=1.5, label="fitted median at site centre")
    fz = flagged["z_depth" if args.vertical == "z_depth" else "elevation"]
    ax.scatter(flagged["su"], fz, s=90, facecolor="none", edgecolor="red", lw=1.5,
               label=f"sample outlier ({len(flagged)})")
    ax.scatter(obs["su"], obs[args.vertical], marker="D", s=70, c="k",
               label="observed target su")
    for _, rr in obs.iterrows():
        ax.annotate(f"{rr['target']} z={rr['z_score']:.1f}", (rr["su"], rr[args.vertical]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("$s_u$ (kPa)")
    ax.set_ylabel(zlab)
    if args.vertical == "z_depth":
        ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Spline outliers: {label}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{prefix}_profile.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(h, tstud, c=cook, s=12, cmap="plasma")
    fig.colorbar(sc, ax=ax, label="Cook's distance")
    for s in (-tcrit, tcrit):
        ax.axhline(s, color="red", ls="--", lw=0.8)
    ax.set_xlabel("leverage h_ii")
    ax.set_ylabel("studentized residual t_i")
    ax.set_title(f"Residual vs leverage (red: Bonferroni |t| = {tcrit:.2f})", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{prefix}_leverage.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    lim = max(bcrit, bhs["z_mean"].abs().max())
    sc = ax.scatter(bhs["x"], bhs["y"], c=bhs["z_mean"], s=60, cmap="coolwarm", vmin=-lim, vmax=lim, edgecolor="k")
    fig.colorbar(sc, ax=ax, label="borehole LOBO z (mean error / sd)")
    for _, rr in bhs[bhs["outlier"]].iterrows():
        ax.annotate(str(rr["borehole_id"]), (rr["x"], rr["y"]), xytext=(5, 5), textcoords="offset points",
                    color="red", weight="bold")
    tb = targets.groupby("borehole_id").first()
    ax.scatter(tb["x"], tb["y"], marker="*", s=200, c="w", edgecolor="k", label="target boreholes")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8)
    ax.set_title(f"Borehole outliers (red labels: |z| > {bcrit:.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{prefix}_boreholes.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved results/{prefix}_*.csv and figures/{prefix}_*.png")


if __name__ == "__main__":
    main()
