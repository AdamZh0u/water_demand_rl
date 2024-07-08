import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import random
import datetime
from src.const import PATH_DATA

# Function to generate water demand time series using ARIMA
def generate_water_demand(n_points=17620+30*24*4, seed=None):
    # (17620 - 100) / 15min /24hours = 182.5 days 
    if seed is not None:
        np.random.seed(seed)
    # ARIMA parameters
    p, d, q = 5, 2, 5
    model = ARIMA(np.random.randn(n_points), order=(p, d, q))
    model_fit = model.fit()
    water_demand = model_fit.predict(start=0, end=n_points-1)
    return water_demand


# Function to inject leakage events into the time series
def inject_leakages(time_series, num_leaks, leak_duration_range=(4, 12), leak_increase_range=(4,8), seed=None):
    if seed is not None:
        random.seed(seed)
    leakage_labels = np.zeros(len(time_series))
    n_points = len(time_series)

    for _ in range(num_leaks):
        leak_start = random.randint(0, n_points - leak_duration_range[1] - 1)
        leak_duration = random.randint(
            leak_duration_range[0], leak_duration_range[1])
        leak_increase = random.uniform(
            leak_increase_range[0], leak_increase_range[1])
        time_series[leak_start:leak_start +
                    leak_duration] *= (1 + leak_increase)
        leakage_labels[leak_start:leak_start + leak_duration] = 1
    return time_series[100:], leakage_labels[100:]


def simulation(seed=50, num_leaks=10):

    # Generate the water demand time series with a fixed seed and number of leakages
    water_demand = generate_water_demand(seed=seed)

    # Inject leakage events with a fixed seed
    water_demand_with_leaks, leakage_labels = inject_leakages(
        water_demand.copy(), num_leaks=num_leaks, seed=seed)

    # Create a time index
    start_date = datetime.datetime.strptime('2024-01-01 00:00', '%Y-%m-%d %H:%M')
    time_index = pd.date_range(start=start_date, periods=len(
        water_demand_with_leaks), freq='15T')

    # Create the DataFrame
    df = pd.DataFrame({
        'Timestamp': time_index,
        'WaterDemandWithLeaks': water_demand_with_leaks,
        'LeakageLabel': leakage_labels
    })
    return df


def plot(df):
    # Plotting the generated data
    plt.figure(figsize=(15, 6))
    plt.plot(df['Timestamp'], df['WaterDemandWithLeaks'],
             label='Water Demand with Leaks', alpha=0.75)
    plt.scatter(df[df['LeakageLabel'] == 1]['Timestamp'], df[df['LeakageLabel']
                == 1]['WaterDemandWithLeaks'], color='red', label='Leakages')
    plt.legend()
    plt.xlabel('Timestamp')
    plt.ylabel('Normalized Water Demand')
    plt.title('Simulated Water Demand with Leakages')


def save(df, seed, num_leaks):
    # Save to Excel
    file_name = 'Data_water_demand_with_leaks_' + \
        str(seed)+'_'+str(num_leaks)+'.txt'
    df.to_csv(PATH_DATA/'simulation'/file_name, index=False)
    print(f'Data saved successfully', file_name)


def load(seed, num_leaks):
    # Load from Excel
    file_name = 'Data_water_demand_with_leaks_' + \
        str(seed)+'_'+str(num_leaks)+'.txt'

    df = pd.read_csv(PATH_DATA/'simulation'/file_name)
    return df

if __name__ == '__main__':
    seed = 50
    num_leaks = 10
    df = simulation(seed, num_leaks)
    plot(df)
    save(df, seed, num_leaks)
