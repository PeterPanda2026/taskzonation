r"""
Penalized-spline (P-spline) prediction of su at the five target boreholes,
using ONLY the coordinates (x, y, depth z) as explanatory variables.

Model (fitted to the 46 training boreholes, Cluster-BID/4)
    ln su = s_z(z) + s_x(x) + s_y(y) + e          (additive in the coordinates)
  * each s_c is a B-spline of degree d on MANY equally spaced segments
    (default 40 in depth, 10 in x and in y), so the basis is deliberately too
    flexible.  Instead of choosing the number of pieces (as spline_select.py
    does) the wiggliness is controlled by difference penalties on
    neighbouring B-spline coefficients (Eilers & Marx):
        minimise ||y - X b||^2 + lambda_z ||D b_z||^2
                               + lambda_h (||D b_x||^2 + ||D b_y||^2)
    D = k-th order difference matrix (default k = 2).  lambda -> 0 gives an
    interpolating spline, lambda -> inf a polynomial of degree k-1 in that
    coordinate (a straight line for k = 2).
  * one weight for depth (lambda_z) and one shared horizontal weight
    (lambda_h) for x and y.
  * a tiny ridge on the x and y coefficients fixes the intercept, which the
    three bases would otherwise share.

Choosing (lambda_z, lambda_h) on a log-spaced 2-D grid (--criterion, default cv)
    gcv : n RSS / (n - edf)^2
    aic : n ln(RSS/n) + 2 (edf + 1)
    cv  : leave-one-borehole-out cross-validation MSE of ln su
  edf = trace of the hat matrix = effective number of parameters.  GCV and
  AIC treat residuals as independent, but residuals within a borehole are
  vertically correlated, so their choice is not reliable for this data.
  Leaving out whole boreholes mimics predicting a new borehole and is the
  default.  On this data GCV/AIC push lambda_h to the low end of any grid
  (unsmoothed x/y effects that fit each borehole), while CV picks a very
  large lambda_h (x and y effects shrunk to straight lines).

The fitted residuals are then used exactly as in spline_select.py: their
vertical correlation rho(dz) = (1 - g) exp(-2|dz|/theta_v) is estimated within
training boreholes, and the residual at the one observed su of each target
borehole updates its predictions (skip with --no-update).  This uses the
measured su itself, not another explanatory variable.

Prediction intervals for a new measured value (Bayesian P-spline variance):
    var = sigma^2 (1 + x' (X'X + P)^-1 x),  sigma^2 = RSS / (n - edf)
back-transformed to su as median, mean and 90% interval.

Outputs
    results/spline_penalized_lambda.csv       GCV, AIC, CV and edf on the lambda grid
    results/spline_penalized_predictions.csv  predictions at all target depths
    results/spline_penalized_correlation.csv  residual correlogram and fitted rho
    figures/spline_penalized_criteria.png     CV / GCV / edf over (lambda_z, lambda_h)
    figures/spline_penalized_fit.png          fitted depth, x and y effects
    figures/spline_penalized_B1..B5.png       per-target predictions

The hidden (shaded) su values of the target boreholes are NEVER used to make a
prediction.  --check scores the predictions against them afterwards and writes
to *_check files, which are git-ignored because they show the hidden values.

Usage
    python spline_penalized.py                      # chosen by CV
    python spline_penalized.py --criterion gcv
    python spline_penalized.py --segments 60 --segments-h 15 --diff-order 3
    python spline_penalized.py --no-update
    python spline_penalized.py --check
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

from spline_select import (COORDS, FIG_DIR, RES_DIR, HERE, bspline_basis, common_args, finish, load,
                           loo_borehole_cv, save)

RIDGE_H = 1e-6      # tiny ridge on the x and y coefficients (identifiability)


class PSpline:
    def __init__(self, segments, degree, diff_order, bounds):
        self.segments, self.degree, self.bounds = segments, degree, bounds
        self.knots = {c: np.linspace(*bounds[c], segments[c] + 1)[1:-1] for c in COORDS}
        sizes = [segments[c] + degree for c in COORDS]
        self.slices = dict(zip(COORDS, np.split(np.arange(sum(sizes)), np.cumsum(sizes)[:-1])))
        p = sum(sizes)
        self.Pz, self.Ph, self.R = np.zeros((p, p)), np.zeros((p, p)), np.zeros((p, p))
        for c in COORDS:
            s = self.slices[c]
            D = np.diff(np.eye(len(s)), n=diff_order, axis=0)
            target = self.Pz if c == "z" else self.Ph
            target[np.ix_(s, s)] = D.T @ D
            if c != "z":
                self.R[s, s] = RIDGE_H

    def penalty(self, lam_z, lam_h):
        return lam_z * self.Pz + lam_h * self.Ph + self.R

    def design(self, df):
        return np.column_stack([bspline_basis(df[c], self.knots[c], self.degree, self.bounds[c]) for c in COORDS])

    def setup(self, df):
        self.X = self.design(df)
        self.y = df["_y"].to_numpy()
        self.XtX, self.Xty = self.X.T @ self.X, self.X.T @ self.y
        self.n = len(self.y)
        return self

    def solve(self, lam_z, lam_h):
        Ainv = np.linalg.inv(self.XtX + self.penalty(lam_z, lam_h))
        coef = Ainv @ self.Xty
        r = self.y - self.X @ coef
        return coef, float(r @ r), float(np.trace(Ainv @ self.XtX)), Ainv

    def fit(self, df, lam_z, lam_h):
        self.setup(df)
        self.lam = (lam_z, lam_h)
        self.coef, self.rss, self.edf, self.Ainv = self.solve(lam_z, lam_h)
        self.resid = self.y - self.X @ self.coef
        self.sigma2 = self.rss / (self.n - self.edf)
        return self

    def predict(self, df):
        X = self.design(df)
        mu = X @ self.coef
        var = self.sigma2 * (1 + np.einsum("ij,jk,ik->i", X, self.Ainv, X))
        return mu, var


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    ap.add_argument("--segments", type=int, default=40, help="equally spaced segments in depth")
    ap.add_argument("--segments-h", type=int, default=10, help="equally spaced segments in x and in y")
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--diff-order", type=int, default=2, help="order of the difference penalty")
    ap.add_argument("--lambda-z", type=float, nargs=3, default=[-2, 5, 22], metavar=("LOG10_MIN", "LOG10_MAX", "N"))
    ap.add_argument("--lambda-h", type=float, nargs=3, default=[-1, 8, 19], metavar=("LOG10_MIN", "LOG10_MAX", "N"))
    ap.add_argument("--criterion", default="cv", choices=["gcv", "aic", "cv"])
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):      # absent in some IDE consoles
        sys.stdout.reconfigure(line_buffering=True)
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    prefix = "spline_penalized"

    train, targets, bounds = load(args.vertical)
    segs = {"z": args.segments, "x": args.segments_h, "y": args.segments_h}
    print(f"Training: {train['borehole_id'].nunique()} boreholes, {len(train)} su values; "
          f"explanatory variables: x, y, {args.vertical}; segments z/x/y = {segs['z']}/{segs['x']}/{segs['y']}, "
          f"degree {args.degree}, difference order {args.diff_order}")

    # 1. choose (lambda_z, lambda_h)
    model = PSpline(segs, args.degree, args.diff_order, bounds).setup(train)
    lz = np.logspace(args.lambda_z[0], args.lambda_z[1], int(args.lambda_z[2]))
    lh = np.logspace(args.lambda_h[0], args.lambda_h[1], int(args.lambda_h[2]))
    pairs = list(itertools.product(lz, lh))
    n = model.n
    rows = []
    for a, b in pairs:
        _, rss, edf, _ = model.solve(a, b)
        rows.append({"lambda_z": a, "lambda_h": b, "edf": edf, "sigma_ln": math.sqrt(rss / (n - edf)),
                     "gcv": n * rss / (n - edf) ** 2, "aic": n * math.log(rss / n) + 2 * (edf + 1)})
    sel = pd.DataFrame(rows)
    print(f"Leave-one-borehole-out CV on {len(pairs)} (lambda_z, lambda_h) pairs ...")
    sel["cv"] = loo_borehole_cv(model.X, model.y, train["borehole_id"].to_numpy(),
                                [model.penalty(a, b) for a, b in pairs])
    for c in ["gcv", "aic", "cv"]:
        b = sel.loc[sel[c].idxmin()]
        edge = []
        for name, v, grid in [("lambda_z", b.lambda_z, lz), ("lambda_h", b.lambda_h, lh)]:
            if v == grid[0]:
                edge.append(f"{name} at the low edge")
            elif v == grid[-1]:
                edge.append(f"{name} at the high edge")
        note = f"  [{', '.join(edge)} of the grid]" if edge else ""
        print(f"  {c.upper():>3} picks lambda_z = {b.lambda_z:.3g}, lambda_h = {b.lambda_h:.3g} "
              f"(edf {b.edf:.1f}){note}")
        sel[f"best_{c}"] = sel.index == b.name
    sel.to_csv(RES_DIR / f"{prefix}_lambda.csv", index=False)
    print("  saved", (RES_DIR / f"{prefix}_lambda.csv").relative_to(HERE))

    best = sel.loc[sel[args.criterion].idxmin()]
    model = PSpline(segs, args.degree, args.diff_order, bounds).fit(train, best.lambda_z, best.lambda_h)
    label = (f"P-spline, lambda_z={best.lambda_z:.3g}, lambda_h={best.lambda_h:.3g}, "
             f"edf={model.edf:.1f} ({args.criterion.upper()})")
    print(f"Using {args.criterion.upper()} choice: {label}, sigma_ln = {math.sqrt(model.sigma2):.3f}")
    print("  (very large lambda_h = x and y effects shrunk to nearly straight lines)")

    # figure of the selection
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, col, title in [(ax[0], "cv", "LOBO-CV MSE of ln su"), (ax[1], "gcv", "GCV"), (ax[2], "edf", "edf")]:
        Z = sel.pivot(index="lambda_h", columns="lambda_z", values=col)
        cs = a.contourf(np.log10(Z.columns), np.log10(Z.index), Z.values, levels=20,
                        cmap="viridis" if col == "edf" else "viridis_r")
        fig.colorbar(cs, ax=a)
        for c, mk in [("cv", "r*"), ("gcv", "w^"), ("aic", "ws")]:
            b = sel.loc[sel[c].idxmin()]
            a.plot(np.log10(b.lambda_z), np.log10(b.lambda_h), mk, ms=12, mec="k", label=c.upper())
        a.set_xlabel("log10 lambda_z (depth)")
        a.set_ylabel("log10 lambda_h (x, y)")
        a.set_title(title, fontsize=10)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Choosing the penalty weights ({args.criterion.upper()} used)")
    save(fig, f"{prefix}_criteria.png")

    # 2-3. residual correlation, predictions, figures, check
    finish(model, train, targets, prefix, label, args)


if __name__ == "__main__":
    main()
