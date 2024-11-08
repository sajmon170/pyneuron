import os
import pandas as pd


csv = 'training/audi.csv'
output = 'training/audi_transformed.csv'

df = pd.read_csv(csv)
df = df.drop(columns=['model'])

transmission_dummy = pd.get_dummies(df['transmission'], prefix='trns', drop_first=True)
fuel_type_dummy = pd.get_dummies(df['fuelType'], prefix='fuel', drop_first=True)

df.drop(columns=['transmission', 'fuelType'], inplace=True)

df = df.join([transmission_dummy, fuel_type_dummy]).dropna()

df.to_csv(output, index=False)
