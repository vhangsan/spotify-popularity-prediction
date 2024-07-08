import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Read the data from the CSV file into a DataFrame
data = pd.read_csv('popular_spotify_songs.csv')

# Cleaning track name
data['track_name'] = data['track_name'].str.strip()  # Remove leading and trailing whitespace

# Cleaning artist name
data['artist(s)_name'] = data['artist(s)_name'].str.replace(',', ', ')  # Add space after comma
data['artist(s)_name'] = data['artist(s)_name'].str.title()  # Convert to title case

# Cleaning released year, month, and day
data['released_year'] = data['released_year'].astype(int)
data['released_month'] = data['released_month'].astype(int)
data['released_day'] = data['released_day'].astype(int)

# Cleaning spotify
data['in_spotify_playlists'] = data['in_spotify_playlists'].astype(int)
data['in_spotify_charts'] = data['in_spotify_charts'].astype(int)

# Cleaning streams
data['streams'] = pd.to_numeric(data['streams'], errors='coerce')

# Cleaning apple
data['in_apple_playlists'] = data['in_apple_playlists'].astype(int)
data['in_apple_charts'] = data['in_apple_charts'].astype(int)

# Cleaning deezer
data['in_deezer_playlists'] = pd.to_numeric(data['in_deezer_playlists'], errors='coerce')
data['in_deezer_charts'] = data['in_deezer_charts'].astype(int)

# Cleaning shazam
data['in_shazam_charts'] = pd.to_numeric(data['in_shazam_charts'], errors='coerce')

# Cleaning BPM
data['bpm'] = data['bpm'].astype(float)

# Cleaning key, mode
data['key'] = data['key'].astype(str).str.upper()  # Convert to uppercase
data['mode'] = data['mode'].astype(str).str.capitalize()  # Capitalize the first letter

columns_to_clean = ['danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 
                    'instrumentalness_%', 'liveness_%', 'speechiness_%']

# Fill missing values with the median for numerical columns
for col in data.select_dtypes(include=np.number):
    data[col].fillna(data[col].median(), inplace=True)

print("Before One-Hot Encoding and Normalization:")
print(data.head())

key_mode_encoder = OneHotEncoder()
key_mode_encoded = key_mode_encoder.fit_transform(data[['key', 'mode']])

# one-hot encoded
key_mode_categories = key_mode_encoder.categories_
key_mode_feature_names = [f"{col}_{value}" for col, values in zip(['key', 'mode'], key_mode_categories) for value in values]

key_mode_encoded_df = pd.DataFrame(key_mode_encoded.toarray(), columns=key_mode_feature_names)
data = pd.concat([data, key_mode_encoded_df], axis=1)
data.drop(['key', 'mode'], axis=1, inplace=True)

# normalize bpm
scaler = StandardScaler()
data['bpm_normalized'] = scaler.fit_transform(data[['bpm']])
data.drop(['bpm'], axis=1, inplace=True)

print("After One-Hot Encoding and Normalization:")
print(data.head())

key_columns = [col for col in data.columns if 'key_' in col]
print("Columns related to 'key':", key_columns)

mode_columns = [col for col in data.columns if 'mode_' in col]
print("Columns related to 'mode':", mode_columns)

cleaned_file_path = 'preprocessed_popular_spotify_songs.csv'
data.to_csv(cleaned_file_path, index=False)