from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from joblib import dump, load
import pandas as pd
import argparse
import os

def preprocess_and_save_pipeline(data, pipeline_path, output_path):
    target = 'Personality'
    categorical = ['Stage_fear', 'Drained_after_socializing']

    # Bersihkan data
    data = data.dropna().drop_duplicates()

    # Encode target di luar pipeline
    le = LabelEncoder()
    data[target] = le.fit_transform(data[target])

    # Ambil fitur numerik
    numeric = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target in numeric: numeric.remove(target)
    if target in categorical: categorical.remove(target)

    # pipeline untuk fitur
    onehot_transform = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    scaler_transform = Pipeline([
        ('scaler', StandardScaler())
    ])

    preprocessing = ColumnTransformer([
        ('num', scaler_transform, numeric),
        ('onehot', onehot_transform, categorical)
    ])

    # Fit pipeline
    features = data.drop(columns=[target])
    X_transformed = preprocessing.fit_transform(features)

    # Simpan pipeline & label encoder
    dump(preprocessing, pipeline_path)
    dump(le, pipeline_path.replace(".joblib", "_labelencoder.joblib"))

    # Buat DataFrame hasil preprocessing
    encod_cols = preprocessing.named_transformers_['onehot']['encoder'].get_feature_names_out(categorical)
    all_columns = numeric + list(encod_cols)

    preprocess_df = pd.DataFrame(X_transformed, columns=all_columns)
    preprocess_df[target] = data[target].values
    preprocess_df.to_csv(output_path, index=False)

    print(f"Pipeline saved to {pipeline_path}")
    print(f"Preprocessed data saved to {output_path}")
    return preprocess_df

def load_and_transform(new_data, pipeline_path):
    target = 'Personality'

    # Load pipeline & label encoder
    preprocessing = load(pipeline_path)
    le = load(pipeline_path.replace(".joblib", "_labelencoder.joblib"))

    # Simpan target (jika ada di data baru)
    if target in new_data.columns:
        new_data[target] = le.transform(new_data[target])
        features = new_data.drop(columns=[target])
    else:
        features = new_data

    # Transform pakai pipeline yang sudah fit
    X_new = preprocessing.transform(features)
    return X_new

# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path ke file input csv mentah")
    parser.add_argument("--pipeline_path", type=str, required=True, help="Path file joblib pipeline")
    parser.add_argument("--output_path", type=str, required=True, help="Path csv hasil preprocessing")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    preprocess_and_save_pipeline(df, args.pipeline_path, args.output_path)
