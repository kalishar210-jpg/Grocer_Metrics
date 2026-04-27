
"""
GrocerMetrics — forecasting.py
Standalone script: loads data, fits ARIMA, generates all charts.

Usage:
    python forecasting.py
    python forecasting.py --steps 60 --out static
"""

import argparse
import logging
import os
import sys
import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from visualizations import (
    ARIMA_ORDER,
    T,
    _clean_data,
    _daily_sales,
    _save,
    _style,
    _title,
    _lerp_hex,
    generate_forecast_image,
    generate_top_selling_products_images,
)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



def load_data(path: str = "Online_Retail.xlsx") -> pd.DataFrame:
    if not os.path.exists(path):
        logger.error("Data file not found: %s", path)
        sys.exit(1)
    logger.info("Loading data from %s …", path)
    df = pd.read_excel(path)
    logger.info("  %d rows, %d columns loaded.", *df.shape)
    logger.info("  Columns: %s", df.columns.tolist())
    return df



def fit_and_diagnose(daily: pd.Series) -> object:
    """Fit ARIMA, print summary, return fitted model."""
    logger.info("Fitting ARIMA%s on %d daily observations …", ARIMA_ORDER, len(daily))

    try:
        model_fit = ARIMA(daily, order=ARIMA_ORDER).fit()
    except Exception as e:
        logger.error("ARIMA fitting failed: %s", e)
        raise

    print("\n" + "=" * 60)
    print(model_fit.summary())
    print("=" * 60 + "\n")

    logger.info("AIC: %.2f   BIC: %.2f", model_fit.aic, model_fit.bic)
    return model_fit


def generate_diagnostics_plot(model_fit, output_dir: str):
    """Save a 2×2 residual diagnostics chart (styled dark theme)."""
    import numpy as np
    from scipy import stats

    residuals = model_fit.resid

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor(T["bg"])
    fig.suptitle("ARIMA Residual Diagnostics", color=T["text"],
                 fontsize=14, fontweight="bold", y=1.01)

    
    ax = axes[0, 0]
    _style(fig, ax)
    ax.plot(residuals.index, residuals.values, color=T["accent"], linewidth=0.8, alpha=0.85)
    ax.axhline(0, color=T["muted"], linewidth=0.8, linestyle="--")
    _title(ax, "Residuals over Time")

    ax = axes[0, 1]
    _style(fig, ax)
    ax.hist(residuals, bins=40, color=T["accent"], alpha=0.6, edgecolor="none", density=True)
    xs = np.linspace(residuals.min(), residuals.max(), 200)
    kde_y = stats.gaussian_kde(residuals)(xs)
    ax.plot(xs, kde_y, color=T["accent2"], linewidth=1.8)
    _title(ax, "Residual Distribution")

    ax = axes[1, 0]
    _style(fig, ax)
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, color=T["accent"], s=10, alpha=0.6, edgecolors="none")
    fit_line = [slope * x + intercept for x in osm]
    ax.plot(osm, fit_line, color=T["accent2"], linewidth=1.5)
    ax.set_xlabel("Theoretical Quantiles", labelpad=6)
    ax.set_ylabel("Sample Quantiles", labelpad=6)
    _title(ax, "Q-Q Plot")

    ax = axes[1, 1]
    _style(fig, ax)
    n_lags = 30
    acf_vals = [residuals.autocorr(lag=i) for i in range(1, n_lags + 1)]
    lags = list(range(1, n_lags + 1))
    ax.bar(lags, acf_vals, color=T["accent"], width=0.6, edgecolor="none")
    ci = 1.96 / (len(residuals) ** 0.5)
    ax.axhline(ci,  color=T["muted"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(-ci, color=T["muted"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(0,   color=T["border"], linewidth=0.6)
    ax.set_xlabel("Lag", labelpad=6)
    _title(ax, "ACF of Residuals", f"95% CI: ±{ci:.3f}")

    fig.tight_layout(pad=2.0)
    path = os.path.join(output_dir, "arima_diagnostics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=T["bg"])
    plt.close(fig)
    logger.info("Diagnostics chart saved → %s", path)



def parse_args():
    p = argparse.ArgumentParser(
        description="GrocerMetrics — generate forecast & product charts"
    )
    p.add_argument("--data",  default="Online_Retail.xlsx", help="Path to Excel data file")
    p.add_argument("--steps", type=int, default=30,         help="Forecast horizon in days (default: 30)")
    p.add_argument("--out",   default="static",             help="Output directory for PNG files")
    p.add_argument("--diag",  action="store_true",          help="Also save ARIMA diagnostics chart")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    
    data = load_data(args.data)

    if args.diag:
        daily = _daily_sales(data)
        model_fit = fit_and_diagnose(daily)
        generate_diagnostics_plot(model_fit, args.out)

    forecast_path = os.path.join(args.out, "sales_forecast.png")
    logger.info("Generating forecast chart (%d steps) …", args.steps)
    generate_forecast_image(data, forecast_steps=args.steps, image_path=forecast_path)

    
    logger.info("Generating product charts …")
    generate_top_selling_products_images(data, static_folder=args.out)

    logger.info("All done. Files written to ./%s/", args.out)


if __name__ == "__main__":
    main()
