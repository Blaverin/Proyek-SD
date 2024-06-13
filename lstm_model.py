import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def prepare_data(file_path, year):
    df = pd.read_csv(file_path, encoding='latin1', dtype={'incident_id': str, 'district_id': str, 'precinct_id': str})
    df['first_occurrence_date'] = pd.to_datetime(df['first_occurrence_date'])
    df = df.dropna()
    df_year = df[df['first_occurrence_date'].dt.year == year]
    return df_year

def lstm_forecast(data, category, steps=30):
    category_data = data[data['offense_category_id'] == category]
    if category_data.empty:
        return [0] * steps  # Jika tidak ada data untuk kategori ini, kembalikan daftar nol
    df_grouped = category_data.groupby('first_occurrence_date').size().reset_index(name='incident_count')

    # Skalakan data menggunakan MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_grouped['incident_count'].values.reshape(-1, 1))

    look_back = 5  # Anda bisa menyesuaikan ini
    if len(scaled_data) <= look_back:
        raise ValueError("Not enough data to train the model")

    X, Y = [], []
    for i in range(len(scaled_data) - look_back - 1):
        a = scaled_data[i:(i + look_back), 0]
        X.append(a)
        Y.append(scaled_data[i + look_back, 0])
    
    if not X or not Y:
        print("Data tidak cukup setelah scaling dan pembuatan X, Y")
        print(f"Scaled data: {scaled_data}")
        raise ValueError("Generated input arrays are empty")
    
    X = np.array(X)
    Y = np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Membangun model LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    model.fit(X, Y, epochs=20, batch_size=1, verbose=2, callbacks=[early_stopping])

    lstm_input = scaled_data[-look_back:]
    lstm_input = np.reshape(lstm_input, (1, look_back, 1))
    forecast = []
    for _ in range(steps):
        prediction = model.predict(lstm_input)
        forecast.append(prediction[0, 0])
        lstm_input = np.append(lstm_input[:, 1:, :], [[[prediction[0, 0]]]], axis=1)
    
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast.flatten().tolist()
