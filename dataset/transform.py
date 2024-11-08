import os
import pandas as pd
from mapping import mapping


directory = 'training'
dataframes = []

wrong_name = 'tax(£)'

for file in os.listdir(directory):
    name = os.path.splitext(file)[0]
    df = pd.read_csv(os.path.join(directory, file))
    if wrong_name in df.columns:
        df.rename(columns={wrong_name: 'tax'}, inplace=True)
    
    df['model'] = df['model'].map(mapping[name])
    df['brand'] = name
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

transmission_dummy = pd.get_dummies(combined_df['transmission'], prefix='trns')
fuel_type_dummy = pd.get_dummies(combined_df['fuelType'], prefix='fuel')
brand_dummy= pd.get_dummies(combined_df['brand'])

combined_df.drop(columns=['transmission', 'fuelType', 'brand'], inplace=True)
combined_df = combined_df.join([transmission_dummy, fuel_type_dummy, brand_dummy]).dropna()

combined_df.to_csv(os.path.join(directory, 'training.csv'), index=False)
print(f"Read {len(combined_df.index)} rows.")
