import os
import pandas as pd
from tqdm import tqdm

from geopy.geocoders import Nominatim


loc_app = Nominatim(user_agent="tutorial")
data_load_path = os.path.join('..', '..', 'data', 'train.csv')
data_dump_path = os.path.join('..', '..', 'data', 'df_for_plot_on_map.csv')


def recognize_location(location: str):
    try:
        return loc_app.geocode(location).raw
    except:
        return None


def prepare_df_for_plot_on_map(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    df = df[df['location'].notna()]
    
    if 'size' not in df.columns:
        df['target'] = df['target'].map({0: 'fake', 1: 'real'})
        df['size'] = 1

    right_locations = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        right_locations.append(recognize_location(row['location']))
        
    df['lat'] = [el['lat'] if isinstance(el, dict) else None for el in right_locations]
    df['lon'] = [el['lon'] if isinstance(el, dict) else None for el in right_locations]
    
    df = df.dropna(subset=['location', 'lat', 'lon'])
    return df


if __name__ == '__main__':
    df = pd.read_csv(data_load_path)
    print(f"DF SHAPE: {df.shape}")

    df = prepare_df_for_plot_on_map(df)
    print(f"PREPARED DF SHAPE: {df.shape}")
    df.to_csv(data_dump_path, index=False)