import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib

data = pd.read_excel('Online_Retail.xlsx')
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
daily_sales = data.groupby('InvoiceDate')['Quantity'].sum().asfreq('D', fill_value=0)
model = ARIMA(daily_sales, order=(1, 1, 1))
model_fit = model.fit()
joblib.dump(model_fit, 'sales_forecast_model.pkl')