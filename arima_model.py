import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def prepare_data(file_path, year):
    df = pd.read_csv(file_path, encoding='latin1', dtype={'incident_id': str, 'district_id': str, 'precinct_id': str})
    df['first_occurrence_date'] = pd.to_datetime(df['first_occurrence_date'])
    df = df.dropna()
    df_year = df[df['first_occurrence_date'].dt.year == year]
    return df_year

def arima_forecast(data, category, steps=30):
    category_data = data[data['offense_category_id'] == category]
    if category_data.empty:
        return [0] * steps  
    df_grouped = category_data.groupby('first_occurrence_date').size().reset_index(name='incident_count')
    model = ARIMA(df_grouped['incident_count'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast.tolist()