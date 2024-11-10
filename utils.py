import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

numerical_columns = ['danceability', 'energy', 'loudness', 'tempo', 'valence', 'mode',
                     'key', 'instrumentalness', 'acousticness', 'speechiness', 'liveness']

def load_and_clean_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(subset=['track_id', 'track_name'], keep='first')
    df['track_name'] = df['track_name'].fillna('')
    df = df[df['track_name'].str.match(r'^[\x00-\x7F]+$')]
    essential_categorical_cols = ['track_name', 'track_genre', 'artists', 'album_name']
    df = df.dropna(subset=essential_categorical_cols)
    df = df[(df['tempo'] >= 60) & (df['tempo'] <= 200)]
    df = df[(df['loudness'] >= -60) & (df['loudness'] <= 0)]
    df = df[df['popularity'] > 1]
    
    num_features = ['duration_ms', 'popularity', 'danceability', 'energy', 'key', 
                    'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
                    'liveness', 'valence', 'tempo', 'time_signature']
    df[num_features] = df[num_features].fillna(df[num_features].median())
    df = pd.get_dummies(df, columns=['explicit'], drop_first=True)
    
    return df

def preprocess_features(df):
    num_features = ['danceability', 'energy', 'loudness', 'tempo', 'valence', 'mode',
                    'key', 'instrumentalness', 'acousticness', 'speechiness', 'liveness']
    
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])
    
    if 'track_genre' in df.columns:
        df_genres = pd.get_dummies(df['track_genre'], prefix='genre')
        df = pd.concat([df, df_genres], axis=1).drop(columns=['track_genre'])
    
    return df, scaler

def recommend_songs(df_combined, song_name, scaler, top_n=5):
    if song_name not in df_combined['track_name'].values:
        return None

    song_data = df_combined[df_combined['track_name'] == song_name].iloc[0]
    song_features = song_data[numerical_columns].values.reshape(1, -1)
    song_features_scaled = scaler.transform(pd.DataFrame(song_features, columns=numerical_columns))
    numerical_features = df_combined[numerical_columns].values
    similarities = cosine_similarity(song_features_scaled, numerical_features)[0]
    df_combined['similarity'] = similarities
    recommended_songs = df_combined[df_combined['track_name'] != song_name].sort_values(by='similarity', ascending=False).head(top_n)

    return recommended_songs[['track_name', 'artists', 'album_name', 'similarity']]