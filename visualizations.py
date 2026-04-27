
"""
GrocerMetrics — visualizations.py
"""

import logging
import warnings
import matplotlib
matplotlib.use('Agg')

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


T = {
    "bg":       "#0e0f11",
    "surface":  "#16181c",
    "surface2": "#1e2026",
    "border":   "#2a2d35",
    "accent":   "#d4f23a",   
    "accent2":  "#3af2c4",   
    "muted":    "#6b7280",
    "text":     "#e8e9ea",
}

ARIMA_ORDER = (5, 1, 0)


def _lerp_hex(c1: str, c2: str, t: float) -> str:
    """Linearly interpolate between two hex colours (0 → c1, 1 → c2)."""
    def parse(h):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r1, g1, b1 = parse(c1)
    r2, g2, b2 = parse(c2)
    return "#{:02x}{:02x}{:02x}".format(
        int(r1 + (r2 - r1) * t),
        int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t),
    )


def _style(fig, ax):
    """Apply dark theme to a figure / axes pair."""
    fig.patch.set_facecolor(T["bg"])
    ax.set_facecolor(T["surface"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(T["border"])
    ax.tick_params(colors=T["muted"], labelsize=9)
    ax.xaxis.label.set_color(T["muted"])
    ax.yaxis.label.set_color(T["muted"])
    ax.grid(True, color=T["surface2"], linewidth=0.7, linestyle="--", alpha=0.9)
    ax.set_axisbelow(True)


def _title(ax, main: str, sub: str = ""):
    ax.set_title(main, color=T["text"], fontsize=13, fontweight="bold", loc="left", pad=14)
    if sub:
        ax.text(0, 1.045, sub, transform=ax.transAxes,
                color=T["muted"], fontsize=8.5, va="bottom")


def _save(fig, path: str):
    fig.tight_layout(pad=1.8)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=T["bg"])
    plt.close(fig)
    logger.info("Saved → %s", path)


def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls, parse dates, keep positive quantities."""
    df = data.copy()
    df.dropna(subset=["InvoiceDate", "Quantity"], inplace=True)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df[df["Quantity"] > 0]


def _daily_sales(data: pd.DataFrame) -> pd.Series:
    return (
        _clean_data(data)
        .groupby("InvoiceDate")["Quantity"]
        .sum()
        .asfreq("D", fill_value=0)
    )

def generate_forecast_image(data: pd.DataFrame, forecast_steps: int, image_path: str):
    """Fit ARIMA and save a styled forecast chart."""
    daily = _daily_sales(data)

    try:
        model_fit = ARIMA(daily, order=ARIMA_ORDER).fit()
    except Exception as e:
        logger.error("ARIMA fit failed: %s", e)
        raise

    # Forecast + confidence interval
    fc_result = model_fit.get_forecast(steps=forecast_steps)
    fc_mean   = fc_result.predicted_mean
    conf_int  = fc_result.conf_int()

    fc_index = pd.date_range(
        start=daily.index.max() + pd.Timedelta(days=1),
        periods=forecast_steps,
        freq="D",
    )
    fc_mean.index  = fc_index
    conf_int.index = fc_index


    fig, ax = plt.subplots(figsize=(14, 5))
    _style(fig, ax)

    # Show only last 180 days of history so the forecast region is readable
    history = daily.iloc[-180:]
    ax.plot(history.index, history.values,
            color=T["accent"], linewidth=1.5, alpha=0.9, label="Historical Sales")

    ax.plot(fc_mean.index, fc_mean.values,
            color=T["accent2"], linewidth=2, linestyle="--",
            label=f"{forecast_steps}-Day Forecast")

    ax.fill_between(
        fc_index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        color=T["accent2"],
        alpha=0.12,
        label="95% Confidence Interval",
    )

    
    ax.axvline(fc_index[0], color=T["border"], linewidth=1, linestyle=":", alpha=0.8)
    ax.text(fc_index[0], ax.get_ylim()[1] * 0.96, "  forecast →",
            color=T["muted"], fontsize=8, va="top")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.set_xlabel("Date", labelpad=8)
    ax.set_ylabel("Units Sold", labelpad=8)

    ax.legend(
        frameon=True,
        facecolor=T["surface2"],
        edgecolor=T["border"],
        labelcolor=T["text"],
        fontsize=8.5,
        loc="upper left",
    )

    _title(ax, f"Sales Forecast — Next {forecast_steps} Days",
           f"ARIMA{ARIMA_ORDER} · {forecast_steps} steps ahead")
    _save(fig, image_path)


def generate_top_selling_products_images(data: pd.DataFrame, static_folder: str, top_n: int = 10):
    """Generate a bar chart and pie chart for the top-N selling products."""
    try:
        df = _clean_data(data)
        top = (
            df.groupby("Description")["Quantity"]
            .sum()
            .nlargest(top_n)
            .sort_values(ascending=True)   
        )

        n = len(top)
        gradient = [_lerp_hex(T["accent"], T["accent2"], i / max(n - 1, 1)) for i in range(n)]

        
        fig, ax = plt.subplots(figsize=(12, 6))
        _style(fig, ax)

        bars = ax.barh(top.index, top.values, color=gradient, height=0.6, edgecolor="none")

        for bar, val in zip(bars, top.values):
            ax.text(
                val + top.max() * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{int(val):,}",
                va="center", color=T["muted"], fontsize=8,
            )

        ax.set_xlabel("Total Units Sold", labelpad=8)
        ax.tick_params(axis="y", labelsize=8, colors=T["text"])
        ax.set_xlim(0, top.max() * 1.15)
        _title(ax, f"Top {top_n} Selling Products", "by total quantity sold")
        _save(fig, f"{static_folder}/top_10_products.png")

        
        pie_colors = [_lerp_hex(T["accent"], T["accent2"], i / max(n - 1, 1))
                      for i in range(n - 1, -1, -1)]

        fig, ax = plt.subplots(figsize=(9, 9))
        fig.patch.set_facecolor(T["bg"])

        wedges, _, autotexts = ax.pie(
            top.values,
            autopct="%1.1f%%",
            startangle=140,
            colors=pie_colors,
            wedgeprops={"linewidth": 1.8, "edgecolor": T["bg"]},
            pctdistance=0.78,
        )
        for at in autotexts:
            at.set_color(T["bg"])
            at.set_fontsize(8)
            at.set_fontweight("bold")

        labels = [t[:38] + "…" if len(t) > 38 else t for t in top.index[::-1]]
        ax.legend(wedges, labels,
                  loc="center left", bbox_to_anchor=(1.0, 0.5),
                  frameon=False, labelcolor=T["text"], fontsize=8)

        ax.set_title(f"Sales Distribution · Top {top_n} Products",
                     color=T["text"], fontsize=13, fontweight="bold", pad=20)

        _save(fig, f"{static_folder}/top_10_products_pie.png")

    except Exception as e:
        logger.error("Error generating product charts: %s", e)
        raise
