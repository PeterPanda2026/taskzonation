r"""
Conditional random field (CRF) prediction of su at unexplored locations.

Step 1 - prior (from rf_mle.py)
    ln su(x, y, z) is a Gaussian random field with
        mean      m(z)   = f(z)^T beta          polynomial trend in depth
        variance  sigma^2
        corr.     rho(tau_h, tau_v; theta_h, theta_v)   separable, exp or sqexp
        nugget    g  (share of sigma^2 that is measurement noise / micro-scale)
    All of these are maximum-likelihood estimates from the 46 training
    boreholes (Cluster-BID/4 = Local-BID-V/4).  They are cached in
    results/crf_prior_*.json so the slow fit runs only once (--refit to redo).

Step 2 - conditioning (simple kriging with the MLE trend as the known mean)
    Observations y_o = ln su at all training samples plus the ONE observed su
    in each target borehole.  For the prediction points u:
        mu_c    = m_u + C_uo C_oo^-1 (y_o - m_o)
        Sigma_c = C_uu - C_uo C_oo^-1 C_ou
    with C = sigma^2 [(1-g) R + g I].  Predictions are for a new measured value
    (nugget included); use --smooth to predict the underlying field instead.
    The autocorrelation is what lets nearby data pull the prediction away from
    the trend and shrink its variance; far from data it falls back to the prior.

Step 3 - lognormal back-transform
    median = exp(mu_c), mean = exp(mu_c + s_c^2/2),
    std = mean * sqrt(exp(s_c^2) - 1), 90% interval = exp(mu_c -/+ 1.645 s_c)

Outputs
    results/crf_predictions.csv        prior and updated statistics at every target depth
    figures/crf_target_B1..B5.png      prior band vs. conditional band + realisations
    figures/crf_map_<depth|elevation><d>m.png   plan maps of conditional median and COV
    results/crf_prior_*.json           cached MLE prior (regenerated with --refit)

The hidden (shaded) su values of the target boreholes are NEVER used to make a
prediction.  --check only reads them afterwards to score the result.  Because
its outputs show the hidden values, --check writes to separate files that are
git-ignored:
    results/crf_predictions_check.csv  as above plus the hidden su column
    figures/crf_target_B1..B5_check.png
    results/crf_check.csv              scores against the hidden su values

Usage
    python crf_predict.py                       # prior fitted on all 46 boreholes
    python crf_predict.py --prior-data cluster  # separate prior per cluster Cn for target Bn
    python crf_predict.py --check               # also score against hidden values
    python crf_predict.py --degree 2 --corr sqexp --refit
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from plot_prediction_data import TARGETS, load_bid, load_targets
from rf_mle import RandomFieldMLE, correlation, trend_basis

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
RES_DIR = HERE / "results"
Z90 = 1.6449
JITTER = 1e-8


# ----------------------------------------------------------------------------
# prior: MLE of the random field parameters
# ----------------------------------------------------------------------------
def fit_prior(train, args, tag):
    """MLE fit on training su; trend degree chosen by BIC unless --degree is given."""
    cache = RES_DIR / f"crf_prior_{tag}_{args.vertical}_{args.corr}_deg{args.degree if args.degree is not None else 'BIC'}.json"
    if cache.exists() and not args.refit:
        prior = json.loads(cache.read_text())
        print(f"  prior '{tag}': loaded from {cache.name}")
        return prior

    coords = train[["x", "y", "_z"]].to_numpy(float)
    y = np.log(train["su"].to_numpy(float))
    degrees = range(args.max_degree + 1) if args.degree is None else [args.degree]
    probe = RandomFieldMLE(coords, y, 0)
    nugget = probe.n_colocated > 0 or args.nugget
    best = None
    for deg in degrees:
        mdl = RandomFieldMLE(coords, y, deg, "z", args.corr, "estimate", nugget, "ml")
        f = mdl.fit("su")
        print(f"  prior '{tag}' degree {deg}: sigma_ln={math.sqrt(f.sigma2):.3f} "
              f"theta_h={f.theta_h:.3g} m theta_v={f.theta_v:.3g} m nugget={f.nugget:.3f} BIC={f.bic:.1f}")
        if best is None or f.bic < best[0].bic:
            best = (f, mdl)
    f, mdl = best
    prior = {"tag": tag, "n": f.n, "degree": f.degree, "labels": f.labels,
             "beta": f.beta.tolist(), "sigma2": f.sigma2, "theta_h": f.theta_h,
             "theta_v": f.theta_v, "nugget": f.nugget, "corr": args.corr,
             "ranges": [list(map(float, r)) for r in mdl.ranges], "vertical": args.vertical}
    RES_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(prior, indent=2))
    print(f"  prior '{tag}': degree {f.degree} chosen, saved to {cache.name}")
    return prior


def trend(prior, coords):
    F, _ = trend_basis(np.asarray(coords, float), prior["degree"], "z", prior["ranges"])
    return F @ np.asarray(prior["beta"])


def cov(prior, A, B, same=False):
    """Covariance of ln su between point sets A and B (nugget only on the diagonal of C(A, A))."""
    dh = np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1])
    dv = np.abs(A[:, None, 2] - B[None, :, 2])
    s2, g = prior["sigma2"], prior["nugget"]
    C = s2 * (1 - g) * correlation(dh, dv, prior["theta_h"], prior["theta_v"], prior["corr"])
    if same:
        C[np.diag_indices_from(C)] += s2 * g
    return C


# ----------------------------------------------------------------------------
# conditioning
# ----------------------------------------------------------------------------
class Conditioner:
    """Pre-factorises C_oo once, then conditions any set of prediction points."""

    def __init__(self, prior, obs_xyz, obs_y):
        self.prior = prior
        self.xo = np.asarray(obs_xyz, float)
        Coo = cov(prior, self.xo, self.xo, same=True)
        Coo[np.diag_indices_from(Coo)] += JITTER
        self.cf = cho_factor(Coo, lower=True, check_finite=False)
        self.alpha = cho_solve(self.cf, np.asarray(obs_y) - trend(prior, self.xo), check_finite=False)

    def predict(self, xu, full_cov=False, smooth=False):
        xu = np.asarray(xu, float)
        Cuo = cov(self.prior, xu, self.xo)
        m_prior = trend(self.prior, xu)
        mu = m_prior + Cuo @ self.alpha
        W = cho_solve(self.cf, Cuo.T, check_finite=False)          # C_oo^-1 C_ou
        s2, g = self.prior["sigma2"], self.prior["nugget"]
        v_prior = s2 * (1 - g) if smooth else s2
        if full_cov:
            Cuu = cov(self.prior, xu, xu, same=not smooth)
            S = Cuu - Cuo @ W
            return m_prior, mu, S
        var = v_prior - np.einsum("ij,ji->i", Cuo, W)
        return m_prior, mu, np.maximum(var, 0.0)


def lognormal_stats(mu, var):
    s = np.sqrt(var)
    mean = np.exp(mu + var / 2)
    return dict(median=np.exp(mu), mean=mean, std=mean * np.sqrt(np.expm1(var)),
                q05=np.exp(mu - Z90 * s), q95=np.exp(mu + Z90 * s), sd_ln=s)


# ----------------------------------------------------------------------------
# main steps
# ----------------------------------------------------------------------------
def predict_targets(args, train_all, targets):
    obs_t = targets[targets["su_observed"]]
    rows, profiles = [], {}
    priors = {}
    for bh, name in TARGETS.items():
        t = targets[targets["borehole_id"] == bh].sort_values("_z")
        if args.prior_data == "cluster":
            n = int(t["cluster"].iloc[0])
            train = train_all[train_all["cluster"] == n]
            tag = f"C{n}"
        else:
            train, tag = train_all, "all"
        if tag not in priors:
            priors[tag] = fit_prior(train, args, tag)
        prior = priors[tag]

        # conditioning data: training su + the observed su of the target borehole(s)
        own_obs = obs_t if args.prior_data == "all" else obs_t[obs_t["borehole_id"] == bh]
        obs = pd.concat([train[["x", "y", "_z", "su"]], own_obs[["x", "y", "_z", "su"]]])
        cond = Conditioner(prior, obs[["x", "y", "_z"]].to_numpy(float), np.log(obs["su"].to_numpy(float)))

        xu = t[["x", "y", "_z"]].to_numpy(float)
        m0, mu, var = cond.predict(xu, smooth=args.smooth)
        v0 = prior["sigma2"] * (1 - prior["nugget"]) if args.smooth else prior["sigma2"]
        pr = lognormal_stats(m0, np.full_like(m0, v0))
        po = lognormal_stats(mu, var)
        for i, (_, r) in enumerate(t.iterrows()):
            row = {"target": name, "borehole_id": bh, "x": r["x"], "y": r["y"],
                   "z_depth": r["z_depth"], "elevation": r["elevation"],
                   "su_observed_used": bool(r["su_observed"]),
                   "su_obs": r["su"] if r["su_observed"] else np.nan,
                   "prior": tag}
            row.update({f"prior_{k}": v[i] for k, v in pr.items()})
            row.update({f"post_{k}": v[i] for k, v in po.items()})
            if args.check:
                row["su_hidden_true"] = r["su"]
            rows.append(row)

        # dense profile along the borehole for plotting + conditional realisations
        z = np.linspace(min(t["_z"].min(), train["_z"].min()), max(t["_z"].max(), train["_z"].max()), 150)
        xd = np.column_stack([np.full_like(z, t["x"].iloc[0]), np.full_like(z, t["y"].iloc[0]), z])
        m0d, mud, S = cond.predict(xd, full_cov=True, smooth=True)
        L = np.linalg.cholesky(S + 1e-6 * prior["sigma2"] * np.eye(len(z)))
        sims = mud[:, None] + L @ np.random.default_rng(args.seed).standard_normal((len(z), args.n_sims))
        profiles[bh] = dict(z=z, m0=m0d, mu=mud, var=np.diag(S), sims=sims, prior=prior, train=train)

    return pd.DataFrame(rows), profiles, priors


def plot_target(df, prof, targets, bh, args):
    name = TARGETS[bh]
    p, z = prof["prior"], prof["z"]
    v0 = p["sigma2"] * (1 - p["nugget"])
    s0, s = math.sqrt(v0), np.sqrt(prof["var"])
    t = targets[targets["borehole_id"] == bh]
    d = df[df["borehole_id"] == bh]
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    ax.scatter(prof["train"]["su"], prof["train"]["_z"], s=5, c="lightgrey", label="training su")
    ax.fill_betweenx(z, np.exp(prof["m0"] - Z90 * s0), np.exp(prof["m0"] + Z90 * s0),
                     color="tab:blue", alpha=0.12, label="prior 90% band")
    ax.plot(np.exp(prof["m0"]), z, color="tab:blue", lw=1.5, label="prior median (MLE trend)")
    ax.plot(np.exp(prof["sims"][:, :args.n_plot_sims]), z, color="tab:orange", lw=0.5, alpha=0.35)
    ax.fill_betweenx(z, np.exp(prof["mu"] - Z90 * s), np.exp(prof["mu"] + Z90 * s),
                     color="tab:red", alpha=0.18, label="conditional 90% band (smooth field)")
    ax.plot(np.exp(prof["mu"]), z, color="tab:red", lw=2, label="conditional median")
    ax.errorbar(d["post_median"], d["_z"] if "_z" in d else d[args.vertical],
                xerr=[d["post_median"] - d["post_q05"], d["post_q95"] - d["post_median"]],
                fmt="o", ms=3, color="darkred", lw=0.8, label="prediction at test depths (90%, incl. nugget)")
    obs = t[t["su_observed"]]
    ax.scatter(obs["su"], obs["_z"], s=120, c="k", marker="D", zorder=6, label="observed su (used)")
    if args.check:
        hid = t[~t["su_observed"]]
        ax.scatter(hid["su"], hid["_z"], s=35, facecolor="none", edgecolor="k", zorder=6,
                   label="hidden su (check only)")
    ax.plot([], [], color="tab:orange", lw=0.8, label=f"conditional realisations ({args.n_plot_sims})")
    ax.set_xscale("log")
    ax.set_xlabel("$s_u$ (kPa)")
    ax.set_ylabel("Depth (m)" if args.vertical == "z_depth" else "Elevation (m)")
    if args.vertical == "z_depth":
        ax.invert_yaxis()
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower left")
    th = f"theta_h={p['theta_h']:.3g} m, theta_v={p['theta_v']:.3g} m"
    ax.set_title(f"{name} (borehole {bh}): prior '{p['tag']}' vs conditional\n{th}", fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / f"crf_target_{name}{'_check' if args.check else ''}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  saved", path.relative_to(HERE))


def plot_maps(prior, train, targets, args):
    """Plan maps of conditional median and COV of su at fixed depths (prior from all data)."""
    obs_t = targets[targets["su_observed"]]
    obs = pd.concat([train[["x", "y", "_z", "su"]], obs_t[["x", "y", "_z", "su"]]])
    cond = Conditioner(prior, obs[["x", "y", "_z"]].to_numpy(float), np.log(obs["su"].to_numpy(float)))
    pts = pd.concat([train.groupby("borehole_id")[["x", "y"]].first(),
                     targets.groupby("borehole_id")[["x", "y"]].first()])
    pad = 15
    gx = np.linspace(pts.x.min() - pad, pts.x.max() + pad, args.map_n)
    gy = np.linspace(pts.y.min() - pad, pts.y.max() + pad, args.map_n)
    X, Y = np.meshgrid(gx, gy)
    tb = targets.groupby("borehole_id").first()
    for zd in args.map_depths:
        xu = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, zd)])
        m0, mu, var = cond.predict(xu, smooth=True)
        st = lognormal_stats(mu, var)
        cov_post = (st["std"] / st["mean"]).reshape(X.shape)
        cov_prior = math.sqrt(math.expm1(prior["sigma2"] * (1 - prior["nugget"])))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
        for ax, Z, lab, cmap in [(axes[0], st["median"].reshape(X.shape), "conditional median su (kPa)", "viridis"),
                                 (axes[1], cov_post, f"conditional COV (prior COV = {cov_prior:.2f})", "magma_r")]:
            cs = ax.contourf(X, Y, Z, levels=20, cmap=cmap)
            fig.colorbar(cs, ax=ax, label=lab)
            ax.scatter(pts.x, pts.y, s=10, c="w", edgecolor="k", lw=0.4)
            ax.scatter(tb.x, tb.y, s=160, marker="*", c="w", edgecolor="k")
            for bh, r in tb.iterrows():
                ax.annotate(TARGETS[bh], (r.x, r.y), xytext=(6, 4), textcoords="offset points",
                            fontsize=9, weight="bold", color="w")
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
        vlab = "depth" if args.vertical == "z_depth" else "elevation"
        fig.suptitle(f"Conditional random field of su at {vlab} {zd:g} m "
                     f"(prior trend median {math.exp(trend(prior, xu[:1])[0]):.1f} kPa)")
        fig.tight_layout()
        path = FIG_DIR / f"crf_map_{vlab}{zd:g}m.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("  saved", path.relative_to(HERE))


def check(df):
    """Score prior and conditional predictions against the hidden su values."""
    h = df[~df["su_observed_used"]].copy()
    out = []
    for stage in ["prior", "post"]:
        e = np.log(h[f"{stage}_median"]) - np.log(h["su_hidden_true"])
        inside = (h["su_hidden_true"] >= h[f"{stage}_q05"]) & (h["su_hidden_true"] <= h[f"{stage}_q95"])
        for name, g in [("all", h.index)] + [(t, h.index[h["target"] == t]) for t in sorted(h["target"].unique())]:
            out.append({"stage": stage, "target": name, "n": len(g),
                        "rmse_ln": float(np.sqrt(np.mean(e[g] ** 2))),
                        "mape_%": float(100 * np.mean(np.abs(h.loc[g, f"{stage}_median"] / h.loc[g, "su_hidden_true"] - 1))),
                        "cover90_%": float(100 * inside[g].mean()),
                        "mean_sd_ln": float(h.loc[g, f"{stage}_sd_ln"].mean())})
    res = pd.DataFrame(out)
    path = RES_DIR / "crf_check.csv"
    res.to_csv(path, index=False)
    print("\nCheck against hidden su (not used in the prediction):")
    print(res.pivot(index="target", columns="stage", values=["rmse_ln", "cover90_%", "mean_sd_ln"])
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print("  saved", path.relative_to(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior-data", default="all", choices=["all", "cluster"],
                    help="fit the prior on all 46 boreholes, or on cluster Cn for target Bn")
    ap.add_argument("--vertical", default="z_depth", choices=["z_depth", "elevation"])
    ap.add_argument("--corr", default="exp", choices=["exp", "sqexp"])
    ap.add_argument("--degree", type=int, default=None, help="trend degree (default: BIC over 0..max-degree)")
    ap.add_argument("--max-degree", type=int, default=3)
    ap.add_argument("--nugget", action="store_true", help="force a nugget (auto-on if samples are co-located)")
    ap.add_argument("--smooth", action="store_true",
                    help="predict the smooth field instead of a new measured value (drops nugget variance)")
    ap.add_argument("--refit", action="store_true", help="ignore the cached prior and refit by MLE")
    ap.add_argument("--n-sims", type=int, default=200, help="conditional realisations per target")
    ap.add_argument("--n-plot-sims", type=int, default=30)
    ap.add_argument("--map-depths", type=float, nargs="*", default=[10, 25, 40],
                    help="depths (or elevations) for plan maps; empty to skip")
    ap.add_argument("--map-n", type=int, default=60, help="map grid points per side")
    ap.add_argument("--check", action="store_true", help="score against the hidden su values afterwards")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):      # absent in some IDE consoles
        sys.stdout.reconfigure(line_buffering=True)
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)

    targets = load_targets()
    train, _ = load_bid()
    for d in (targets, train):
        d["_z"] = d[args.vertical]
    print(f"Training: {train['borehole_id'].nunique()} boreholes, {len(train)} su values; "
          f"targets: {len(targets)} depths, {targets['su_observed'].sum()} observed su")

    print("\nStep 1-2: MLE prior and conditioning")
    df, profiles, priors = predict_targets(args, train, targets)
    path = RES_DIR / f"crf_predictions{'_check' if args.check else ''}.csv"
    df.to_csv(path, index=False)
    print("  saved", path.relative_to(HERE))

    for tag, p in priors.items():
        print(f"\n  prior '{tag}': degree {p['degree']}, sigma_ln={math.sqrt(p['sigma2']):.3f}, "
              f"theta_h={p['theta_h']:.3g} m, theta_v={p['theta_v']:.3g} m, nugget={p['nugget']:.3f}")
    summ = df.groupby("target").agg(n=("z_depth", "size"), prior_sd_ln=("prior_sd_ln", "mean"),
                                    post_sd_ln=("post_sd_ln", "mean"))
    summ["sd_reduction_%"] = 100 * (1 - summ["post_sd_ln"] / summ["prior_sd_ln"])
    print("\nUncertainty of ln su before / after conditioning (mean over target depths):")
    print(summ.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nStep 3: figures")
    df["_z"] = df[args.vertical]
    for bh in TARGETS:
        plot_target(df, profiles[bh], targets, bh, args)
    if args.map_depths:
        if args.prior_data == "all":
            plot_maps(priors["all"], train, targets, args)
        else:
            print("  plan maps need one site-wide prior; run with --prior-data all")

    if args.check:
        check(df)


if __name__ == "__main__":
    main()
