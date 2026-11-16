r"""
Random field parameter estimation by maximum likelihood (MLE) for the
undrained shear strength su.  Only su is modelled; LL, PL and w in the data
files are not used.

Model
-----
X = su is lognormal, so Y = ln su is Gaussian:

    Y(x, y, z) = f(x, y, z)^T beta + eps(x, y, z)

  * f^T beta : polynomial trend of degree p
               --trend-vars z    ->  1, z, z^2, ..., z^p          (default)
               --trend-vars xyz  ->  all monomials x^i y^j z^k with i+j+k <= p
               (coordinates are scaled to [-1, 1] before building the basis)
  * eps      : zero-mean stationary Gaussian field, variance sigma^2, with a
               separable correlation function written in terms of the
               horizontal / vertical scales of fluctuation (dh, dv):
                 exp    (Markov):  rho = exp(-2|tau_h|/dh) * exp(-2|tau_v|/dv)
                 sqexp  (Gauss) :  rho = exp(-pi (tau_h/dh)^2) * exp(-pi (tau_v/dv)^2)
               tau_h = horizontal distance, tau_v = vertical distance.
  * optional nugget (measurement error): C = sigma^2 [(1-g) R + g I], 0 <= g < 1.

With --horizontal independent the boreholes are treated as uncorrelated
(rho = 0 between boreholes), and only dv is estimated.

Likelihood
----------
For fixed correlation parameters theta = (dh, dv[, g]) the GLS estimates of
beta and sigma^2 have closed forms, so the "concentrated" (profile) negative
log-likelihood depends on theta only:

  ML  : beta = (F'K^-1 F)^-1 F'K^-1 y ,  sigma^2 = r'K^-1 r / n
        -lnL = n/2 ln(2 pi sigma^2) + 1/2 ln|K| + n/2
  REML: sigma^2 = r'K^-1 r / (n-k)
        -lnL = (n-k)/2 ln(2 pi sigma^2) + 1/2 ln|K| + 1/2 ln|F'K^-1 F| + (n-k)/2

theta is found by multi-start L-BFGS-B in log space.  The log-likelihood of
the original (lognormal) data is lnL_Y - sum(ln x_i); AIC/BIC are computed
from the ML (not REML) value and are used to choose the trend degree.

Usage
-----
    python rf_mle.py                                          # su in the 46 training boreholes
    python rf_mle.py --max-degree 2 --select aic
    python rf_mle.py --horizontal independent --corr sqexp --nugget
    python rf_mle.py --file BM1_five_borehole_data.xlsx       # only 5 visible su values: very uncertain
    python rf_mle.py --selftest                               # recover known parameters from simulated data
"""
import argparse
import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.optimize import minimize
from scipy.special import expit

PROPS = ["su"]              # the only property estimated
JITTER = 1e-8
COLOCATED_TOL = 0.05       # m; samples closer than this count as co-located


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_data(path, sheet=0, keep_unobserved=False, max_boreholes=None, seed=0):
    df = pd.read_excel(path, sheet_name=sheet)
    # A text column may flag rows as "unobserved data": in the target-borehole
    # file this marks su values that are hidden (kept for checking only), so
    # they are blanked out and never used.
    extra = [c for c in df.columns if str(c).startswith("Unnamed")]
    if extra and not keep_unobserved:
        flag = df[extra].astype(str).apply(lambda s: s.str.contains("unobserved", case=False)).any(axis=1)
        if flag.any():
            print(f"{flag.sum()} su value(s) flagged 'unobserved' are left out")
            df.loc[flag, "su"] = np.nan
    df = df.dropna(subset=["borehole_id", "x", "y", "z_depth"])
    if max_boreholes and df.borehole_id.nunique() > max_boreholes:
        ids = np.random.default_rng(seed).choice(df.borehole_id.unique(), max_boreholes, replace=False)
        df = df[df.borehole_id.isin(ids)]
        print(f"Using a random subset of {max_boreholes} boreholes ({len(df)} samples)")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# model pieces
# ----------------------------------------------------------------------------
def scale(v, lo, hi):
    return np.zeros_like(v) if hi == lo else 2 * (v - lo) / (hi - lo) - 1


def trend_basis(coords, degree, trend_vars, ranges):
    """Polynomial design matrix F (n x k) with column labels."""
    names = {"x": 0, "y": 1, "z": 2}
    idx = [names[c] for c in trend_vars]
    s = np.column_stack([scale(coords[:, i], *ranges[i]) for i in idx])
    terms = [p for p in itertools.product(range(degree + 1), repeat=len(idx)) if sum(p) <= degree]
    terms.sort(key=sum)                                   # constant, linear, quadratic, ...
    cols = [np.prod([s[:, j] ** pw for j, pw in enumerate(p)], axis=0) for p in terms]
    labels = ["*".join(f"{trend_vars[j]}^{pw}" for j, pw in enumerate(p) if pw) or "1" for p in terms]
    return np.column_stack(cols), labels


def correlation(dh, dv, theta_h, theta_v, kind):
    """Separable correlation. theta_h=None -> independent boreholes (dh>0 gives 0)."""
    if kind == "exp":
        arg = (-2.0 / theta_v) * dv
        if theta_h is not None:
            arg += (-2.0 / theta_h) * dh
    else:
        arg = (-np.pi / theta_v ** 2) * dv ** 2
        if theta_h is not None:
            arg += (-np.pi / theta_h ** 2) * dh ** 2
    R = np.exp(arg, out=arg)
    if theta_h is None:
        R[dh > 0] = 0.0
    return R


@dataclass
class Fit:
    prop: str
    degree: int
    method: str
    n: int
    k: int
    beta: np.ndarray
    beta_se: np.ndarray
    labels: list
    sigma2: float
    theta_h: float
    theta_v: float
    nugget: float
    nll: float                 # negative log-lik of ln X under `method`
    nll_ml: float              # ML negative log-lik of ln X (for AIC/BIC)
    loglik_x: float            # ML log-lik of the original lognormal data
    n_par: int
    ci: dict = field(default_factory=dict)

    @property
    def aic(self):
        return 2 * self.n_par - 2 * self.loglik_x

    @property
    def bic(self):
        return self.n_par * math.log(self.n) - 2 * self.loglik_x


class RandomFieldMLE:
    def __init__(self, coords, y, degree=1, trend_vars="z", corr="exp",
                 horizontal="estimate", nugget=False, method="ml"):
        self.coords = np.asarray(coords, float)
        self.y = np.asarray(y, float)
        self.n = len(y)
        self.corr, self.method = corr, method
        self.est_h = horizontal == "estimate"
        self.est_g = nugget
        xyz = self.coords
        self.ranges = [(xyz[:, i].min(), xyz[:, i].max()) for i in range(3)]
        self.F, self.labels = trend_basis(xyz, degree, trend_vars, self.ranges)
        self.k = self.F.shape[1]
        self.degree = degree
        self.dh = np.hypot(xyz[:, None, 0] - xyz[None, :, 0], xyz[:, None, 1] - xyz[None, :, 1])
        self.dv = np.abs(xyz[:, None, 2] - xyz[None, :, 2])
        sep = np.hypot(self.dh, self.dv)
        np.fill_diagonal(sep, np.inf)
        self.n_colocated = int((sep < COLOCATED_TOL).sum() // 2)
        zr = max(np.ptp(xyz[:, 2]), 1.0)
        hr = max(self.dh.max(), 1.0)
        # Below the closest borehole spacing the likelihood is flat in theta_h (no
        # pairs to inform it), so the lower bound is half that spacing.
        pos = self.dh[self.dh > 1e-6]
        hmin = 0.5 * pos.min() if pos.size else 0.5
        self.bounds_v = (0.02, 5 * zr)
        self.bounds_h = (hmin, 20 * hr)

    # ---- parameter packing (optimise in log / logit space) ----
    def unpack(self, p):
        i = 0
        th_h = None
        if self.est_h:
            th_h = math.exp(p[i]); i += 1
        th_v = math.exp(p[i]); i += 1
        g = 1 / (1 + math.exp(-p[i])) if self.est_g else 0.0
        return th_h, th_v, g

    def K(self, th_h, th_v, g):
        R = correlation(self.dh, self.dv, th_h, th_v, self.corr)
        K = (1 - g) * R
        K[np.diag_indices_from(K)] += g + JITTER
        return K

    def gls(self, th_h, th_v, g):
        """Closed-form beta, sigma^2 and the pieces of the likelihood for given theta."""
        cf = cho_factor(self.K(th_h, th_v, g), lower=True, check_finite=False)
        logdetK = 2 * np.sum(np.log(np.diag(cf[0])))
        KiF = cho_solve(cf, self.F, check_finite=False)
        A = self.F.T @ KiF
        beta = np.linalg.solve(A, KiF.T @ self.y)
        r = self.y - self.F @ beta
        q = r @ cho_solve(cf, r, check_finite=False)
        return beta, q, logdetK, A

    def nll_theta(self, th_h, th_v, g, method=None):
        method = method or self.method
        beta, q, logdetK, A = self.gls(th_h, th_v, g)
        n, k = self.n, self.k
        if method == "ml":
            s2 = q / n
            return 0.5 * n * math.log(2 * math.pi * s2) + 0.5 * logdetK + 0.5 * n
        s2 = q / (n - k)
        _, logdetA = np.linalg.slogdet(A)
        return 0.5 * (n - k) * math.log(2 * math.pi * s2) + 0.5 * logdetK + 0.5 * logdetA + 0.5 * (n - k)

    def objective(self, p):
        try:
            return self.nll_theta(*self.unpack(p))
        except (LinAlgError, ValueError, FloatingPointError):
            return 1e12

    def fit(self, prop="", n_polish=3):
        """Coarse grid scan of the concentrated likelihood, then L-BFGS-B polishing.

        Starting points are the best grid cell for each of `n_polish` different
        first-parameter values, so the starts are spread out rather than clustered
        on one flat region of the likelihood.
        """
        lv = np.log(self.bounds_v)
        lh = np.log(self.bounds_h)
        grids = [np.linspace(lh[0] + 0.2, lh[1] - 0.5, 7)] if self.est_h else []
        grids.append(np.linspace(lv[0] + 0.3, lv[1] - 0.3, 8))
        if self.est_g:
            grids.append(np.array([-4.0, -1.5, 0.5]))
        bounds = ([tuple(lh)] if self.est_h else []) + [tuple(lv)] + ([(-9.0, 4.0)] if self.est_g else [])
        best_per_level = {}
        for s in itertools.product(*grids):
            s = np.array(s)
            f = self.objective(s)
            if s[0] not in best_per_level or f < best_per_level[s[0]][0]:
                best_per_level[s[0]] = (f, s)
        ranked = sorted(best_per_level.values(), key=lambda t: t[0])
        starts = [s for _, s in ranked[:n_polish]]
        best = None
        for s in starts:
            res = minimize(self.objective, np.array(s), method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        p = best.x
        th_h, th_v, g = self.unpack(p)
        beta, q, logdetK, A = self.gls(th_h, th_v, g)
        s2 = q / (self.n if self.method == "ml" else self.n - self.k)
        beta_se = np.sqrt(np.diag(s2 * np.linalg.inv(A)))
        nll_ml = self.nll_theta(th_h, th_v, g, "ml")
        n_par = self.k + 1 + len(p)                       # beta, sigma^2, theta
        sum_lnx = self.y.sum()
        fit = Fit(prop, self.degree, self.method, self.n, self.k, beta, beta_se, self.labels,
                  s2, th_h, th_v, g, best.fun, nll_ml, -nll_ml - sum_lnx, n_par)
        fit.ci = self.profile_ci(p, bounds)
        return fit

    def profile_ci(self, p, bounds, h=1e-3):
        """Approximate 95% CI for theta from the numerical Hessian in log space."""
        m = len(p)
        H = np.zeros((m, m))
        f0 = self.objective(p)
        for i in range(m):
            for j in range(i, m):
                ei, ej = np.eye(m)[i] * h, np.eye(m)[j] * h
                if i == j:
                    H[i, i] = (self.objective(p + ei) - 2 * f0 + self.objective(p - ei)) / h ** 2
                else:
                    H[i, j] = H[j, i] = (self.objective(p + ei + ej) - self.objective(p + ei - ej)
                                         - self.objective(p - ei + ej) + self.objective(p - ei - ej)) / (4 * h ** 2)
        names = (["theta_h"] if self.est_h else []) + ["theta_v"] + (["nugget"] if self.est_g else [])
        out = {}
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return out
        for i, nm in enumerate(names):
            at_bound = min(abs(p[i] - bounds[i][0]), abs(p[i] - bounds[i][1])) < 1e-3
            if not np.isfinite(cov[i, i]) or cov[i, i] <= 0 or at_bound:
                out[nm] = None
                continue
            se = math.sqrt(cov[i, i])
            lo, hi = p[i] - 1.96 * se, p[i] + 1.96 * se
            # A nearly flat likelihood gives a huge se; the interval is then
            # clipped to the search bounds and reported as open-ended.
            lo_c, hi_c = max(lo, bounds[i][0]), min(hi, bounds[i][1])
            tf = expit if nm == "nugget" else np.exp
            out[nm] = (float(tf(lo_c)), float(tf(hi_c)), lo < bounds[i][0], hi > bounds[i][1])
        return out


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def fmt_ci(ci, key):
    v = ci.get(key)
    if v is None:
        return "  (95% CI n/a)"
    lo, hi, lo_open, hi_open = v
    return f"  (95% CI {'<' if lo_open else ''}{lo:.3g} - {'>' if hi_open else ''}{hi:.3g})"


def report(fit, args):
    print(f"\n--- {fit.prop}: degree {fit.degree}, {fit.method.upper()}, n = {fit.n} ---")
    for lab, b, se in zip(fit.labels, fit.beta, fit.beta_se):
        print(f"  beta[{lab:>12}] = {b: .4f}  (se {se:.4f})")
    sd = math.sqrt(fit.sigma2)
    print(f"  sigma_lnX       = {sd:.4f}   ->  COV of X = {math.sqrt(math.exp(fit.sigma2) - 1):.3f}")
    if fit.theta_h is not None:
        print(f"  theta_h (m)     = {fit.theta_h:.3g}" + fmt_ci(fit.ci, "theta_h"))
    else:
        print("  theta_h         = 0 (boreholes treated as independent)")
    print(f"  theta_v (m)     = {fit.theta_v:.3g}" + fmt_ci(fit.ci, "theta_v"))
    if args.nugget:
        print(f"  nugget ratio    = {fit.nugget:.3f}" + fmt_ci(fit.ci, "nugget"))
    print(f"  lnL (X, ML)     = {fit.loglik_x:.2f}   AIC = {fit.aic:.2f}   BIC = {fit.bic:.2f}")


def plot_fit(df, model, fit, args, out):
    z = df["_z"].to_numpy()
    zz = np.linspace(z.min(), z.max(), 200)
    c = np.column_stack([np.full_like(zz, df.x.mean()), np.full_like(zz, df.y.mean()), zz])
    F, _ = trend_basis(c, fit.degree, args.trend_vars, model.ranges)
    m = F @ fit.beta
    sd = math.sqrt(fit.sigma2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    ax = axes[0]
    ax.plot(np.log(df[fit.prop]), z, ".", alpha=0.5, label="data")
    ax.plot(m, zz, "k-", label="trend (at mean x, y)" if "x" in args.trend_vars else "trend")
    ax.fill_betweenx(zz, m - 2 * sd, m + 2 * sd, color="k", alpha=0.12, label="trend +/- 2 sigma")
    ax.set_xlabel(f"ln {fit.prop}")
    ax.set_ylabel(args.depth_col)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(df[fit.prop], z, ".", alpha=0.5)
    ax.plot(np.exp(m), zz, "k-", label="median")
    ax.plot(np.exp(m + fit.sigma2 / 2), zz, "r--", label="mean")
    ax.plot(np.exp(m - 2 * sd), zz, "k:", lw=1)
    ax.plot(np.exp(m + 2 * sd), zz, "k:", lw=1)
    ax.set_xlabel(fit.prop)
    ax.legend(fontsize=8)
    if args.depth_col == "z_depth":
        axes[0].invert_yaxis()
    th = "indep." if fit.theta_h is None else f"{fit.theta_h:.3g} m"
    fig.suptitle(f"{fit.prop}: degree-{fit.degree} trend, sigma_ln={sd:.3f}, "
                 f"theta_h={th}, theta_v={fit.theta_v:.3g} m ({args.corr}, {fit.method.upper()})")
    fig.tight_layout()
    path = out / f"mle_{fit.prop}_fit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  saved", path)


def plot_surface(model, fit, out, ngrid=30):
    """Concentrated -lnL over (theta_h, theta_v), or a 1-D profile if theta_h is not estimated."""
    g = fit.nugget
    tv = np.geomspace(*model.bounds_v, ngrid)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if model.est_h:
        th = np.geomspace(*model.bounds_h, ngrid)
        Z = np.array([[model.nll_theta(a, b, g) for a in th] for b in tv])
        cs = ax.contourf(th, tv, Z - Z.min(), levels=np.r_[0, 1, 2, 3, 5, 10, 20, 50, 100, 200], cmap="viridis_r")
        fig.colorbar(cs, ax=ax, label="-lnL - min(-lnL)")
        ax.plot(fit.theta_h, fit.theta_v, "r*", ms=14, label="MLE")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("theta_h (m)"); ax.set_ylabel("theta_v (m)")
        ax.legend()
    else:
        nll = [model.nll_theta(None, b, g) for b in tv]
        ax.plot(tv, nll, "-")
        ax.axvline(fit.theta_v, color="r", ls="--", label="MLE")
        ax.set_xscale("log"); ax.set_xlabel("theta_v (m)"); ax.set_ylabel("-lnL")
        ax.legend()
    ax.set_title(f"{fit.prop}: likelihood surface")
    fig.tight_layout()
    path = out / f"mle_{fit.prop}_likelihood.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  saved", path)


# ----------------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------------
def run(args):
    df = load_data(args.file, args.sheet, args.keep_unobserved, args.max_boreholes, args.seed)
    df["_z"] = df[args.depth_col]
    print(f"File: {args.file.name}   boreholes: {df.borehole_id.nunique()}   samples: {len(df)}")
    if len(df) > 3000:
        print("Warning: n > 3000 - each likelihood call needs an n x n Cholesky; "
              "consider --max-boreholes.")
    args.out.mkdir(parents=True, exist_ok=True)
    xyz = df[["x", "y", "_z"]].to_numpy(float)
    probe = RandomFieldMLE(xyz, np.zeros(len(xyz)), 0)
    if probe.n_colocated and not args.nugget:
        print(f"Note: {probe.n_colocated} pair(s) of samples are co-located (< {COLOCATED_TOL} m apart) "
              "with different values. Without a nugget the correlation matrix is singular, "
              "so the nugget is switched on.")
        args.nugget = True
    del probe
    rows = []
    for prop in PROPS:
        d = df[df[prop] > 0].reset_index(drop=True)
        if len(d) < len(df):
            print(f"{prop}: dropped {len(df) - len(d)} non-positive / missing values")
        if len(d) < 20:
            print(f"Warning: only {len(d)} {prop} values - too few for a reliable random-field fit "
                  f"(estimates and intervals will be very uncertain)")
        coords = d[["x", "y", "_z"]].to_numpy()
        y = np.log(d[prop].to_numpy())
        fits, models = [], []
        for deg in range(args.max_degree + 1):
            mdl = RandomFieldMLE(coords, y, deg, args.trend_vars, args.corr, args.horizontal,
                                 args.nugget, args.method)
            if mdl.k > len(d) - 3:
                print(f"{prop:>3} degree {deg}: skipped ({mdl.k} trend terms for {len(d)} values)")
                continue
            f = mdl.fit(prop)
            fits.append(f); models.append(mdl)
            th = "indep" if f.theta_h is None else f"{f.theta_h:8.3g}"
            print(f"{prop:>3} degree {deg}: sigma_ln={math.sqrt(f.sigma2):.3f}  theta_h={th}  "
                  f"theta_v={f.theta_v:6.3g}  AIC={f.aic:9.2f}  BIC={f.bic:9.2f}")
        if not fits:
            print(f"{prop}: not enough values to fit any trend degree - skipped")
            continue
        if args.degree is None:
            i = int(np.argmin([getattr(f, args.select) for f in fits]))
        else:
            i = next((j for j, f in enumerate(fits) if f.degree == args.degree), None)
            if i is None:
                print(f"{prop}: degree {args.degree} could not be fitted - skipped")
                continue
        best, mdl = fits[i], models[i]
        report(best, args)
        plot_fit(d, mdl, best, args, args.out)
        if len(d) <= 1500:
            plot_surface(mdl, best, args.out)
        rows.append({"property": prop, "n": best.n, "degree": best.degree,
                     **{f"beta_{lab}": b for lab, b in zip(best.labels, best.beta)},
                     "sigma_lnX": math.sqrt(best.sigma2),
                     "COV_X": math.sqrt(math.exp(best.sigma2) - 1),
                     "theta_h_m": best.theta_h, "theta_v_m": best.theta_v,
                     "nugget": best.nugget, "loglik_X": best.loglik_x,
                     "AIC": best.aic, "BIC": best.bic})
    table = pd.DataFrame(rows)
    path = args.out / f"mle_results_{args.file.stem}.csv"
    table.to_csv(path, index=False)
    print(f"\nSummary written to {path}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def selftest(args):
    """Simulate a lognormal field with known parameters on the cluster-file geometry and refit."""
    path = args.file.parent / "BM1_Cluster_BID_4_minus_5BHs.xlsx"
    df = load_data(path) if path.exists() else load_data(args.file)
    coords = df[["x", "y", "z_depth"]].to_numpy(float)
    true = dict(beta=np.array([3.6, 0.8]), sigma=0.35, th_h=40.0, th_v=2.5, g=0.1)
    mdl = RandomFieldMLE(coords, np.zeros(len(coords)), 1, "z", "exp", "estimate", True, "ml")
    L = np.linalg.cholesky(mdl.K(true["th_h"], true["th_v"], true["g"]))
    rng = np.random.default_rng(args.seed)
    print(f"Self-test on {len(coords)} locations; true: beta={true['beta']}, sigma_ln={true['sigma']}, "
          f"theta_h={true['th_h']}, theta_v={true['th_v']}, nugget={true['g']}")
    for rep in range(args.reps):
        y = mdl.F @ true["beta"] + true["sigma"] * (L @ rng.standard_normal(len(coords)))
        m = RandomFieldMLE(coords, y, 1, "z", "exp", "estimate", True, "ml")
        f = m.fit("sim")
        print(f"  rep {rep}: beta={np.round(f.beta, 3)}  sigma_ln={math.sqrt(f.sigma2):.3f}  "
              f"theta_h={f.theta_h:.3g}  theta_v={f.theta_v:.3g}  nugget={f.nugget:.3f}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=here / "BM1_Cluster_BID_4_minus_5BHs.xlsx",
                    help="borehole file (default: the 46 training boreholes next to this script)")
    ap.add_argument("--sheet", default=0)
    ap.add_argument("--depth-col", default="z_depth", choices=["z_depth", "elevation"])
    ap.add_argument("--trend-vars", default="z", choices=["z", "xyz"])
    ap.add_argument("--max-degree", type=int, default=3, help="fit trend degrees 0..max-degree")
    ap.add_argument("--degree", type=int, default=None, help="force this degree instead of AIC/BIC choice")
    ap.add_argument("--select", default="bic", choices=["aic", "bic"])
    ap.add_argument("--corr", default="exp", choices=["exp", "sqexp"])
    ap.add_argument("--horizontal", default="estimate", choices=["estimate", "independent"])
    ap.add_argument("--nugget", action="store_true", help="also estimate a nugget (measurement-error) ratio")
    ap.add_argument("--method", default="ml", choices=["ml", "reml"])
    ap.add_argument("--keep-unobserved", action="store_true", help="keep rows flagged 'unobserved data'")
    ap.add_argument("--max-boreholes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=here / "mle_output")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--reps", type=int, default=3, help="self-test replicates")
    args = ap.parse_args()
    import sys
    if hasattr(sys.stdout, "reconfigure"):      # absent in some IDE consoles
        sys.stdout.reconfigure(line_buffering=True)
    if args.degree is not None:
        args.max_degree = max(args.max_degree, args.degree)
    selftest(args) if args.selftest else run(args)


if __name__ == "__main__":
    main()
