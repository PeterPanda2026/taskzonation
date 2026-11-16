r"""
Regression-spline prediction of su at the five target boreholes, using ONLY the
coordinates (x, y, depth z) as explanatory variables, with the number of spline
pieces chosen by model selection.

Model (fitted to the 46 training boreholes, Cluster-BID/4)
    ln su = s_z(z) + s_x(x) + s_y(y) + e          (additive in the coordinates)
  * each s_c is a B-spline of degree d with K_c pieces (K_c - 1 interior knots
    at quantiles: of the sample depths for z, of the borehole positions for x
    and y; or equally spaced with --knots uniform).  K = 1 is a plain
    degree-d polynomial in that coordinate.
  * K_z >= 1; K_x, K_y >= 0, where 0 drops that coordinate (no trend in it).
  * the z basis carries the intercept; the first column of the x and y bases
    is dropped so the model is identifiable.
  * fitted by ordinary least squares on ln su (su is lognormal).

Choosing the numbers of pieces (--criterion, default cv) over all
(K_z, K_x, K_y) on the grid 1..--max-pieces x 0..--max-pieces-h x 0..--max-pieces-h:
    aic : n ln(RSS/n) + 2 (p + 1)
    bic : n ln(RSS/n) + ln(n) (p + 1)
    cv  : leave-one-borehole-out cross-validation MSE of ln su
  AIC/BIC assume independent residuals, but residuals within a borehole are
  vertically correlated, so their choice is not reliable for this data.
  Leaving out whole boreholes mimics predicting a new borehole and is the
  default.  (The knot positions are fixed from all training data in CV.)

Using the one observed su per target borehole (skip with --no-update)
  This uses the measured su itself, not another explanatory variable.  The
  residuals of the fitted model are vertically correlated; their correlation
  is estimated from training residuals within boreholes as
      rho(dz) = (1 - g) exp(-2 |dz| / theta_v)       (g = nugget share)
  and the residual r0 at the observed depth z0 of each target updates the
  prediction at depth z (simple kriging with one point):
      mean  += (1 - g) exp(-2|z - z0|/theta_v) r0
      var   -= sigma^2 [(1 - g) exp(-2|z - z0|/theta_v)]^2
  Far from z0 (relative to theta_v) the prediction returns to the regression.

Prediction intervals are for a new measured value:
    var = sigma^2 (1 + x' (X'X)^-1 x)   before the update
and are back-transformed: median exp(mu), mean exp(mu + s^2/2),
90% interval exp(mu -/+ 1.645 s).

Outputs
    results/spline_select_models.csv       AIC, BIC, CV for every (K_z, K_x, K_y)
    results/spline_select_predictions.csv  predictions at all target depths
    results/spline_select_correlation.csv  residual correlogram and fitted rho
    figures/spline_select_criteria.png     criteria vs K_z, and CV over (K_x, K_y)
    figures/spline_select_fit.png          fitted depth, x and y effects
    figures/spline_select_B1..B5.png       per-target predictions

The hidden (shaded) su values of the target boreholes are NEVER used to make a
prediction.  --check scores the predictions against them afterwards and writes
to *_check files, which are git-ignored because they show the hidden values.

Usage
    python spline_select.py                         # cubic, chosen by CV
    python spline_select.py --criterion bic
    python spline_select.py --degrees 1 2 3 --max-pieces 20 --max-pieces-h 5
    python spline_select.py --no-update
    python spline_select.py --check
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from plot_prediction_data import TARGETS, load_bid, load_targets

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
RES_DIR = HERE / "results"
Z90 = 1.6449
COORDS = ["z", "x", "y"]            # z = depth (or elevation), x, y = plan coordinates


# ----------------------------------------------------------------------------
# shared pieces (also imported by spline_penalized.py)
# ----------------------------------------------------------------------------
def load(vertical):
    """Training data, target data, and the coordinate ranges the bases must cover."""
    train, _ = load_bid()
    targets = load_targets()
    for d in (train, targets):
        d["z"] = d[vertical].astype(float)
        d["_y"] = np.log(d["su"].astype(float))
    bounds = {}
    for c, pad in [("z", 0.5), ("x", 1.0), ("y", 1.0)]:
        v = np.r_[train[c], targets[c]]
        bounds[c] = (v.min() - pad, v.max() + pad)
    return train, targets, bounds


def bspline_basis(v, knots_inner, degree, bounds):
    """B-spline design matrix with clamped boundary knots at `bounds`."""
    lo, hi = bounds
    t = np.r_[[lo] * (degree + 1), np.sort(knots_inner), [hi] * (degree + 1)]
    return BSpline.design_matrix(np.clip(np.asarray(v, float), lo, hi), t, degree).toarray()


def penalized_solve(XtX, Xty, P=None):
    """Solve (X'X + P) b = X'y with a tiny ridge for numerical safety."""
    A = XtX if P is None else XtX + P
    A = A + 1e-9 * np.trace(A) / len(A) * np.eye(len(A))
    return np.linalg.solve(A, Xty)


def loo_borehole_cv(X, y, groups, penalties=(None,)):
    """Leave-one-borehole-out MSE of ln su for each penalty matrix in
    `penalties` (None = unpenalized least squares), re-using X'X."""
    XtX, Xty = X.T @ X, X.T @ y
    sse = np.zeros(len(penalties))
    for idx in pd.Series(range(len(y))).groupby(np.asarray(groups)).indices.values():
        Xi, yi = X[idx], y[idx]
        XtX_f, Xty_f = XtX - Xi.T @ Xi, Xty - Xi.T @ yi
        for j, P in enumerate(penalties):
            sse[j] += np.sum((yi - Xi @ penalized_solve(XtX_f, Xty_f, P)) ** 2)
    return sse / len(y)


def fit_vertical_correlation(train, resid, max_lag=20.0, bin_width=1.5):
    """Correlogram of residuals within boreholes and least-squares fit of
    rho(dz) = (1-g) exp(-2 dz / theta_v)."""
    s2 = np.mean(resid ** 2)
    lags, prods = [], []
    for idx in train.groupby("borehole_id").indices.values():
        z, r = train["z"].to_numpy()[idx], resid[idx]
        dz = np.abs(z[:, None] - z[None, :])
        iu = np.triu_indices(len(z), 1)
        keep = dz[iu] <= max_lag
        lags.append(dz[iu][keep])
        prods.append((r[:, None] * r[None, :])[iu][keep])
    lags, prods = np.concatenate(lags), np.concatenate(prods)
    edges = np.arange(0, max_lag + bin_width, bin_width)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lags > a) & (lags <= b)
        if m.sum() >= 10:
            rows.append({"lag_m": lags[m].mean(), "pairs": int(m.sum()), "rho_emp": prods[m].mean() / s2})
    cg = pd.DataFrame(rows)

    def res(p):
        g, th = p
        return np.sqrt(cg["pairs"]) * ((1 - g) * np.exp(-2 * cg["lag_m"] / th) - cg["rho_emp"])

    sol = least_squares(res, x0=[0.2, 5.0], bounds=([0.0, 0.1], [0.99, 200.0]))
    g, theta_v = sol.x
    cg["rho_fit"] = (1 - g) * np.exp(-2 * cg["lag_m"] / theta_v)
    return {"sigma2": s2, "nugget": float(g), "theta_v": float(theta_v)}, cg


def update_with_observation(targets, mu, var, model, corr):
    """Shift each target borehole's prediction by the residual at its observed
    su, weighted by the vertical residual correlation."""
    mu, var = mu.copy(), var.copy()
    for bh in TARGETS:
        m = (targets["borehole_id"] == bh).to_numpy()
        obs = targets[m & targets["su_observed"].to_numpy()]
        if obs.empty:
            continue
        r0 = obs["_y"].iloc[0] - model.predict(obs.iloc[[0]])[0][0]
        rho = (1 - corr["nugget"]) * np.exp(-2 * np.abs(targets.loc[m, "z"].to_numpy() - obs["z"].iloc[0])
                                            / corr["theta_v"])
        mu[m] += rho * r0
        var[m] = np.maximum(var[m] - corr["sigma2"] * rho ** 2, 1e-12)
    return mu, var


def lognormal_stats(mu, var):
    s = np.sqrt(var)
    mean = np.exp(mu + var / 2)
    return dict(median=np.exp(mu), mean=mean, std=mean * np.sqrt(np.expm1(var)),
                q05=np.exp(mu - Z90 * s), q95=np.exp(mu + Z90 * s), sd_ln=s)


def predict_and_save(model, train, targets, prefix, args):
    """Residual correlation, predictions at the target depths, CSV outputs."""
    corr, cg = fit_vertical_correlation(train, model.resid)
    cg.to_csv(RES_DIR / f"{prefix}_correlation.csv", index=False)
    print(f"Residual correlation: sigma_ln={math.sqrt(corr['sigma2']):.3f}, "
          f"nugget={corr['nugget']:.2f}, theta_v={corr['theta_v']:.2f} m")
    mu, var = model.predict(targets)
    mu_u, var_u = (mu, var) if args.no_update else update_with_observation(targets, mu, var, model, corr)
    reg, upd = lognormal_stats(mu, var), lognormal_stats(mu_u, var_u)
    rows = []
    for i, (_, r) in enumerate(targets.iterrows()):
        row = {"target": r["target"], "borehole_id": r["borehole_id"], "x": r["x"], "y": r["y"],
               "z_depth": r["z_depth"], "elevation": r["elevation"],
               "su_observed_used": bool(r["su_observed"]), "su_obs": r["su"] if r["su_observed"] else np.nan}
        row.update({f"reg_{k}": v[i] for k, v in reg.items()})
        row.update({f"pred_{k}": v[i] for k, v in upd.items()})
        if args.check:
            row["su_hidden_true"] = r["su"]
        rows.append(row)
    df = pd.DataFrame(rows)
    path = RES_DIR / f"{prefix}_predictions{'_check' if args.check else ''}.csv"
    df.to_csv(path, index=False)
    print("  saved", path.relative_to(HERE))
    return df, corr, cg


def _vaxis(ax, args):
    ax.set_ylabel("Depth (m)" if args.vertical == "z_depth" else "Elevation (m)")
    if args.vertical == "z_depth":
        ax.invert_yaxis()


def save(fig, name):
    fig.tight_layout()
    path = FIG_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  saved", path.relative_to(HERE))


def plot_components(model, train, prefix, label, args):
    """Fitted depth profile at the site centre, and the x and y effects with
    borehole-mean partial residuals."""
    centre = {c: float(train[c].median()) for c in COORDS}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    zg = np.linspace(train["z"].min(), train["z"].max(), 300)
    m, v = model.predict(pd.DataFrame({"z": zg, "x": centre["x"], "y": centre["y"]}))
    ax = axes[0]
    ax.scatter(train["su"], train["z"], s=5, c="lightgrey", label="training su")
    ax.plot(np.exp(m), zg, "b-", lw=2, label="fitted median at site centre")
    ax.fill_betweenx(zg, np.exp(m - Z90 * np.sqrt(v)), np.exp(m + Z90 * np.sqrt(v)), alpha=0.15,
                     label="90% interval")
    ax.set_xscale("log")
    ax.set_xlabel("$s_u$ (kPa)")
    _vaxis(ax, args)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    resid = pd.Series(model.resid, index=train.index)
    bh = train.groupby("borehole_id")[["x", "y"]].first()
    bh["r"] = resid.groupby(train["borehole_id"]).mean()
    for ax, c in zip(axes[1:], ["x", "y"]):
        g = np.linspace(*model.bounds[c], 200)
        grid = pd.DataFrame({**centre, c: g})
        base = model.predict(pd.DataFrame([centre]))[0][0]
        eff = model.predict(grid)[0] - base
        ax.plot(g, eff, "b-", lw=2, label=f"effect of {c} on ln su")
        at = model.predict(pd.DataFrame({**centre, c: bh[c].to_numpy()}))[0] - base
        ax.plot(bh[c], at + bh["r"], "o", ms=4, alpha=0.6, label="borehole partial residual")
        for k in model.knots.get(c, []):
            ax.axvline(k, color="k", ls=":", lw=0.6)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel(f"{c} (m)")
        ax.set_ylabel("change in ln su")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(label, fontsize=10)
    save(fig, f"{prefix}_fit.png")


def plot_correlogram(cg, corr, prefix, label):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cg["lag_m"], cg["rho_emp"], "o", label="empirical (within boreholes)")
    lag = np.linspace(0, cg["lag_m"].max(), 200)
    ax.plot(lag, (1 - corr["nugget"]) * np.exp(-2 * lag / corr["theta_v"]), "k-",
            label=f"fit: (1-{corr['nugget']:.2f}) exp(-2 dz/{corr['theta_v']:.2f} m)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("vertical lag dz (m)")
    ax.set_ylabel("residual correlation")
    ax.set_title(f"Residual vertical correlation\n{label}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, f"{prefix}_correlation.png")


def plot_targets(df, model, train, targets, prefix, label, args):
    sfx = "_check" if args.check else ""
    for bh, name in TARGETS.items():
        t = targets[targets["borehole_id"] == bh]
        d = df[df["borehole_id"] == bh]
        zg = np.linspace(train["z"].min(), train["z"].max(), 300)
        m, v = model.predict(pd.DataFrame({"z": zg, "x": t["x"].iloc[0], "y": t["y"].iloc[0]}))
        s = np.sqrt(v)
        fig, ax = plt.subplots(figsize=(6.5, 7.5))
        ax.scatter(train["su"], train["z"], s=5, c="lightgrey", label="training su")
        ax.fill_betweenx(zg, np.exp(m - Z90 * s), np.exp(m + Z90 * s), color="tab:blue", alpha=0.12,
                         label="regression 90% interval")
        ax.plot(np.exp(m), zg, color="tab:blue", lw=1.5, label="regression median at this (x, y)")
        ax.errorbar(d["pred_median"], t["z"], xerr=[d["pred_median"] - d["pred_q05"], d["pred_q95"] - d["pred_median"]],
                    fmt="o", ms=3, lw=0.8, color="tab:red",
                    label="prediction at target depths (90%)" + ("" if args.no_update else ", updated"))
        obs = t[t["su_observed"]]
        ax.scatter(obs["su"], obs["z"], s=120, c="k", marker="D", zorder=6, label="observed su (used)")
        if args.check:
            hid = t[~t["su_observed"]]
            ax.scatter(hid["su"], hid["z"], s=35, facecolor="none", edgecolor="k", zorder=6,
                       label="hidden su (check only)")
        ax.set_xscale("log")
        ax.set_xlabel("$s_u$ (kPa)")
        _vaxis(ax, args)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_title(f"{name} (borehole {bh}, x={t['x'].iloc[0]:.0f} m, y={t['y'].iloc[0]:.0f} m)\n{label}",
                     fontsize=9)
        save(fig, f"{prefix}_{name}{sfx}.png")


def check(df, prefix):
    """Score regression-only and updated predictions against the hidden su."""
    h = df[~df["su_observed_used"]]
    out = []
    for stage in ["reg", "pred"]:
        e = np.log(h[f"{stage}_median"]) - np.log(h["su_hidden_true"])
        inside = (h["su_hidden_true"] >= h[f"{stage}_q05"]) & (h["su_hidden_true"] <= h[f"{stage}_q95"])
        for name in ["all"] + sorted(h["target"].unique()):
            g = h.index if name == "all" else h.index[h["target"] == name]
            out.append({"stage": "regression" if stage == "reg" else "updated", "target": name, "n": len(g),
                        "rmse_ln": float(np.sqrt(np.mean(e[g] ** 2))),
                        "cover90_%": float(100 * inside[g].mean())})
    res = pd.DataFrame(out)
    path = RES_DIR / f"{prefix}_check.csv"
    res.to_csv(path, index=False)
    print("\nCheck against hidden su (not used in the prediction):")
    print(res.pivot(index="target", columns="stage", values=["rmse_ln", "cover90_%"])
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("  saved", path.relative_to(HERE))


def common_args(ap):
    ap.add_argument("--vertical", default="z_depth", choices=["z_depth", "elevation"],
                    help="vertical coordinate z")
    ap.add_argument("--no-update", action="store_true",
                    help="do not use the observed su of each target borehole")
    ap.add_argument("--check", action="store_true", help="score against the hidden su values afterwards")


def finish(model, train, targets, prefix, label, args):
    df, corr, cg = predict_and_save(model, train, targets, prefix, args)
    plot_components(model, train, prefix, label, args)
    plot_correlogram(cg, corr, prefix, label)
    plot_targets(df, model, train, targets, prefix, label, args)
    if args.check:
        check(df, prefix)


# ----------------------------------------------------------------------------
# regression spline with model selection
# ----------------------------------------------------------------------------
def inner_knots(v, pieces, how):
    if pieces <= 1:
        return np.empty(0)
    q = np.linspace(0, 1, pieces + 1)[1:-1]
    return np.quantile(v, q) if how == "quantile" else np.min(v) + q * np.ptp(v)


class RegressionSpline:
    """Additive B-spline regression in z, x, y; pieces = {"z": Kz, "x": Kx, "y": Ky}."""

    def __init__(self, pieces, degree, knots_how, bounds):
        self.pieces, self.degree, self.knots_how, self.bounds = pieces, degree, knots_how, bounds

    def set_knots(self, df):
        bh = df.groupby("borehole_id")[["x", "y"]].first()
        src = {"z": df["z"].to_numpy(), "x": bh["x"].to_numpy(), "y": bh["y"].to_numpy()}
        self.knots = {c: inner_knots(src[c], self.pieces[c], self.knots_how)
                      for c in COORDS if self.pieces[c] >= 1}
        return self

    def design(self, df):
        cols = []
        for c in COORDS:
            if self.pieces[c] >= 1:
                B = bspline_basis(df[c], self.knots[c], self.degree, self.bounds[c])
                cols.append(B if c == "z" else B[:, 1:])
        return np.column_stack(cols)

    def fit(self, df):
        self.set_knots(df)
        X, y = self.design(df), df["_y"].to_numpy()
        self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.resid = y - X @ self.coef
        self.n, self.p = X.shape
        self.rss = float(self.resid @ self.resid)
        self.sigma2 = self.rss / (self.n - self.p)
        self.XtX_inv = np.linalg.pinv(X.T @ X)
        return self

    def predict(self, df):
        X = self.design(df)
        mu = X @ self.coef
        var = self.sigma2 * (1 + np.einsum("ij,jk,ik->i", X, self.XtX_inv, X))
        return mu, var

    def aic(self):
        return self.n * math.log(self.rss / self.n) + 2 * (self.p + 1)

    def bic(self):
        return self.n * math.log(self.rss / self.n) + math.log(self.n) * (self.p + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    ap.add_argument("--degrees", type=int, nargs="+", default=[3], help="spline degrees to try")
    ap.add_argument("--max-pieces", type=int, default=15, help="try K_z = 1 .. max-pieces")
    ap.add_argument("--max-pieces-h", type=int, default=4, help="try K_x, K_y = 0 .. max-pieces-h")
    ap.add_argument("--knots", default="quantile", choices=["quantile", "uniform"])
    ap.add_argument("--criterion", default="cv", choices=["aic", "bic", "cv"])
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):      # absent in some IDE consoles
        sys.stdout.reconfigure(line_buffering=True)
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    prefix = "spline_select"

    train, targets, bounds = load(args.vertical)
    y, groups = train["_y"].to_numpy(), train["borehole_id"].to_numpy()
    print(f"Training: {train['borehole_id'].nunique()} boreholes, {len(train)} su values; "
          f"explanatory variables: x, y, {args.vertical}")

    # 1. model selection over (degree, K_z, K_x, K_y)
    grid = list(itertools.product(args.degrees, range(1, args.max_pieces + 1),
                                  range(args.max_pieces_h + 1), range(args.max_pieces_h + 1)))
    print(f"Comparing {len(grid)} models ...")
    rows = []
    for d, kz, kx, ky in grid:
        m = RegressionSpline({"z": kz, "x": kx, "y": ky}, d, args.knots, bounds).fit(train)
        rows.append({"degree": d, "K_z": kz, "K_x": kx, "K_y": ky, "n_par": m.p,
                     "sigma_ln": math.sqrt(m.sigma2), "aic": m.aic(), "bic": m.bic(),
                     "cv": loo_borehole_cv(m.design(train), y, groups)[0]})
    sel = pd.DataFrame(rows)
    for c in ["aic", "bic", "cv"]:
        sel[f"best_{c}"] = sel[c] == sel[c].min()
        b = sel.loc[sel[c].idxmin()]
        print(f"  {c.upper():>3} picks degree {int(b.degree)}, K_z={int(b.K_z)}, K_x={int(b.K_x)}, "
              f"K_y={int(b.K_y)}  ({int(b.n_par)} parameters)")
    sel.to_csv(RES_DIR / f"{prefix}_models.csv", index=False)
    print("  saved", (RES_DIR / f"{prefix}_models.csv").relative_to(HERE))
    print(f"Top 5 by {args.criterion.upper()}:")
    print(sel.nsmallest(5, args.criterion)[["degree", "K_z", "K_x", "K_y", "n_par", "sigma_ln", "aic", "bic", "cv"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    best = sel.loc[sel[args.criterion].idxmin()]
    d, pieces = int(best.degree), {"z": int(best.K_z), "x": int(best.K_x), "y": int(best.K_y)}
    model = RegressionSpline(pieces, d, args.knots, bounds).fit(train)
    label = (f"Regression spline, degree {d}, K_z={pieces['z']}, K_x={pieces['x']}, K_y={pieces['y']} "
             f"({args.criterion.upper()})")
    print(f"Using {args.criterion.upper()} choice: {label}")

    # figures of the selection
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    s = sel[(sel.degree == d) & (sel.K_x == pieces["x"]) & (sel.K_y == pieces["y"])]
    ax[0].plot(s.K_z, s.aic - s.aic.min(), "o-", label="AIC - min")
    ax[0].plot(s.K_z, s.bic - s.bic.min(), "s--", label="BIC - min")
    ax0b = ax[0].twinx()
    ax0b.plot(s.K_z, s.cv, "^-", color="tab:red", label="LOBO-CV MSE")
    ax0b.set_ylabel("LOBO-CV MSE of ln su", color="tab:red")
    ax[0].axvline(pieces["z"], color="k", ls=":", lw=1)
    ax[0].set_xlabel("number of depth pieces K_z")
    ax[0].set_ylabel("criterion - min")
    ax[0].set_title(f"vs K_z (K_x={pieces['x']}, K_y={pieces['y']})", fontsize=9)
    ax[0].legend(loc="upper center", fontsize=8)
    ax0b.legend(loc="upper right", fontsize=8)
    ax[0].grid(alpha=0.3)
    h = sel[sel.degree == d].groupby(["K_y", "K_x"])["cv"].min().unstack()
    im = ax[1].imshow(h.values, origin="lower", cmap="viridis_r", aspect="auto")
    fig.colorbar(im, ax=ax[1], label="LOBO-CV MSE (best K_z)")
    ax[1].set_xticks(range(len(h.columns)), h.columns)
    ax[1].set_yticks(range(len(h.index)), h.index)
    ax[1].plot(list(h.columns).index(pieces["x"]), list(h.index).index(pieces["y"]), "r*", ms=15)
    ax[1].set_xlabel("K_x (0 = no x term)")
    ax[1].set_ylabel("K_y (0 = no y term)")
    ax[1].set_title("CV over horizontal pieces", fontsize=9)
    fig.suptitle("Choosing the number of spline pieces")
    save(fig, f"{prefix}_criteria.png")

    # 2-3. residual correlation, predictions, figures, check
    finish(model, train, targets, prefix, label, args)


if __name__ == "__main__":
    main()
