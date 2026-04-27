"""
GrocerMetrics — app.py
Market Basket Analysis & Sales Prediction API
"""
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from visualizations import generate_forecast_image, generate_top_selling_products_images
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

app = Flask(__name__)


sales_model      = joblib.load("grocery_model.pkl")
forecast_model   = joblib.load("sales_forecast_model.pkl")
recommender_model = pd.read_csv("recommender_model.csv")
data             = pd.read_excel("Online_Retail.xlsx")

def _parse_frozensets(df: pd.DataFrame) -> pd.DataFrame:
    """Convert stringified frozensets in antecedents/consequents to real sets."""
    for col in ("antecedents", "consequents"):
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            )
    return df

recommender_model = _parse_frozensets(recommender_model)

product_list = data["Description"].dropna().unique().tolist()



@app.route("/")
def index():
    return render_template("index.html", products=product_list)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        description, quantity, unit_price = _parse_predict_form(request.form)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    try:
        sales_input      = pd.DataFrame([[quantity, unit_price]], columns=["Quantity", "UnitPrice"])
        sales_prediction = sales_model.predict(sales_input)[0]

        days_to_sell         = calculate_days_to_sell(description, quantity)
        purchase_probability = calculate_purchase_probability(description)
        recommendations      = get_recommendations(description)

        result = {
            "sales_prediction":    float(sales_prediction),
            "unit_price":          unit_price,
            "quantity":            quantity,
            "product_description": description,
            "recommendations":     recommendations,
            "days_to_sell":        days_to_sell,
            "purchase_probability": purchase_probability,
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@app.route("/forecast")
def forecast():

    return render_template("forecast.html")


@app.route("/api/forecast-data", methods=["POST"])
def get_forecast_data():
    """Return forecast data as JSON for dynamic chart updates."""
    try:
        
        json_data = request.get_json(force=True, silent=True) or {}
        forecast_days = int(json_data.get("forecastDays", 30))
        if forecast_days < 1 or forecast_days > 365:
            forecast_days = 30
        
        df = data.copy()
        df.dropna(subset=["InvoiceDate", "Quantity"], inplace=True)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df = df[df["Quantity"] > 0]
        
        daily_sales = (
            df.groupby("InvoiceDate")["Quantity"]
            .sum()
            .asfreq("D", fill_value=0)
        )
        
        try:
            model_fit = ARIMA(daily_sales, order=(5, 1, 0)).fit()
        except Exception as e:
            return jsonify({"error": f"Model fitting error: {str(e)}"}), 500
        

        fc_result = model_fit.get_forecast(steps=forecast_days)
        fc_mean = fc_result.predicted_mean
        conf_int = fc_result.conf_int()
        
        
        fc_index = pd.date_range(
            start=daily_sales.index.max() + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D",
        )
        fc_mean.index = fc_index
        conf_int.index = fc_index
        
        
        history = daily_sales.iloc[-180:]
        response = {
            "historical": {
                "dates": [d.strftime("%Y-%m-%d") for d in history.index],
                "values": history.values.tolist()
            },
            "forecast": {
                "dates": [d.strftime("%Y-%m-%d") for d in fc_index],
                "values": fc_mean.values.tolist(),
                "upper": conf_int.iloc[:, 1].values.tolist(),
                "lower": conf_int.iloc[:, 0].values.tolist()
            },
            "forecastDays": forecast_days
        }
        return jsonify(response)
        
    except Exception as e:
        app.logger.error("Forecast data generation failed: %s", str(e))
        return jsonify({"error": f"Forecast error: {str(e)}"}), 500


@app.route("/api/top-products", methods=["GET"])
def p():
    """Return top selling products data as JSON."""
    try:
        top_n = int(request.args.get("topN", 10))
        
        df = data.copy()
        df.dropna(subset=["Description", "Quantity"], inplace=True)
        df = df[df["Quantity"] > 0]
        
        top = (
            df.groupby("Description")["Quantity"]
            .sum()
            .nlargest(top_n)
            .sort_values(ascending=False)
        )
        
        response = {
            "products": top.index.tolist(),
            "quantities": top.values.tolist(),
            "topN": top_n
        }
        return jsonify(response)
        
    except Exception as e:
        app.logger.error("Top products retrieval failed: %s", str(e))
        return jsonify({"error": f"Error: {str(e)}"}), 500



def _parse_predict_form(form) -> tuple:
    """Extract and validate prediction form inputs. Returns (description, quantity, unit_price)."""
    description = form.get("Description", "").strip()
    quantity    = form.get("Quantity", "").strip()
    unit_price  = form.get("UnitPrice", "").strip()

    if not description or not quantity or not unit_price:
        raise ValueError("All fields (Description, Quantity, Unit Price) are required.")

    try:
        quantity   = float(quantity)
        unit_price = float(unit_price)
    except ValueError:
        raise ValueError("Quantity and Unit Price must be valid numbers.")

    if quantity <= 0 or unit_price < 0:
        raise ValueError("Quantity must be positive and Unit Price cannot be negative.")

    return description, quantity, unit_price


def calculate_days_to_sell(product_description: str, quantity: float) -> float:
    """Estimate how many days it will take to sell `quantity` units of a product."""
    product_sales = (
        data[data["Description"] == product_description]
        .groupby("InvoiceDate")["Quantity"]
        .sum()
        .asfreq("D", fill_value=0)
    )
    avg_daily_sales = product_sales.mean()
    if avg_daily_sales > 0:
        return round(quantity / avg_daily_sales, 2)
    return float("inf")


def calculate_purchase_probability(product_description: str) -> float:
    """Return the product's share of total quantity sold across all products."""
    total_sales   = data["Quantity"].sum()
    product_sales = data[data["Description"] == product_description]["Quantity"].sum()
    if total_sales > 0:
        return round(float(product_sales) / float(total_sales), 4)
    return 0.0


def get_recommendations(product_description: str) -> list:
    """Return up to 5 cross-sell recommendations via association rules."""
    if recommender_model.empty:
        return []

    rules = recommender_model[
        recommender_model["antecedents"].apply(lambda x: product_description in x)
    ]

    rules = rules.sort_values(["confidence", "lift"], ascending=[False, False])

    recommendations = []
    for _, row in rules.head(5).iterrows():
        for item in row["consequents"]:
            if item != product_description and item not in recommendations:
                recommendations.append(item)
            if len(recommendations) >= 5:
                break
        if len(recommendations) >= 5:
            break

    return recommendations




if __name__ == "__main__":
    app.run(debug=True)