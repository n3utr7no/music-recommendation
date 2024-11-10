from flask import Flask, request, render_template, jsonify
import pandas as pd
from utils import load_and_clean_dataset, preprocess_features, recommend_songs

app = Flask(__name__)

df_combined = None
scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global df_combined, scaler
    file = request.files['file']
    
    if file:
        file_path = 'uploaded_dataset.csv'
        file.save(file_path)
        
       
        df = load_and_clean_dataset(file_path)
        df_combined, scaler = preprocess_features(df)
        
        return jsonify({"message": "Dataset uploaded and processed successfully!"})
    else:
        return jsonify({"error": "File upload failed"}), 400

@app.route('/recommend', methods=['POST'])
def recommend():
    global df_combined, scaler
    song_name = request.form.get('song_name')
    
    if df_combined is None or scaler is None:
        return jsonify({"error": "Dataset not loaded. Please upload a dataset first."}), 400
    
    recommended_songs = recommend_songs(df_combined, song_name, scaler)
    
    if recommended_songs is not None and not recommended_songs.empty:
        recommendations = recommended_songs.to_dict(orient='records')
        return jsonify({"recommendations": recommendations})
    else:
        return jsonify({"error": "No recommendations found"}), 404

if __name__ == '__main__':
    app.run(debug=True)