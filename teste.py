import pandas as pd
from prophet import Prophet

dado_csv = pd.read_csv('Dados.xlsx - Query result(1).csv')

dado_csv['Data'] = pd.to_datetime(dado_csv['Data'], dayfirst=True)

dado_csv['Dia_da_semana'] = dado_csv['Data'].dt.day_name()

print(dado_csv)

p = Prophet(interval_width=0.92, daily_seasonality=True)

dado_csv.columns = ['ds', 'y', 'dia_semana']
dado_csv.head()

model = p.fit(dado_csv)

future = p.make_future_dataframe(periods=5, freq='D')
future.tail()

forecast = p.predict(future)
forecast[['ds', 'yhat']].tail()
