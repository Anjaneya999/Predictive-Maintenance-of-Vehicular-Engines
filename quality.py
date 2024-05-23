import pandas as pd

df = pd.read_csv('CO2 Emissions_Canada.csv')


def predict_quality(v_class, result):
    df2 = df[df['Vehicle Class'] == v_class]
    fir = df2['CO2 Emissions(g/km)'].mean() + df2['CO2 Emissions(g/km)'].std()
    sec = df2['CO2 Emissions(g/km)'].mean() + 2 * df2['CO2 Emissions(g/km)'].std()
    if result <= fir:
        return "1: The vehicle is functioning without any issues!"
    elif result <= sec:
        return "2: The vehicle is in need of a maintenance check-up!"
    return "3: URGENT!!! The vehicle is releasing hazardous levels of emissions and needs immediate repair!"
