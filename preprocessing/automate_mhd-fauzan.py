from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import argparse
import os

def preprocess_data(data, path):

    # Ambil fitur numerik dan kategorikal
    numeric = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical = ['Stage_fear', 'Drained_after_socializing']
    target = 'Personality'

    # encode kolom target
    le = LabelEncoder()
    data[target] = le.fit_transform(data[target])
    # bersihkan data dari missing value dan duplicate value

    data = data.dropna()
    data = data.drop_duplicates()

    # hindari target masuk ke pipeline
    if target in numeric:
        numeric.remove(target)
    if target in categorical:
        categorical.remove(target)

    # pipeline masing - masing tipe
    onehot_transform = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])


    scaler_transform = Pipeline([
        ('scaler', StandardScaler())
    ])

    prerocessing = ColumnTransformer([
        ('num', scaler_transform, numeric),
        ('onehot', onehot_transform, categorical),
    ])

    transform_data = prerocessing.fit_transform(data)

    encod_cols = prerocessing.named_transformers_['onehot']['encoder'].get_feature_names_out(categorical)

    all_columns = numeric + list(encod_cols)

    # gabung hasil dengan target
    preprocess_df = pd.DataFrame(transform_data, columns=all_columns)
    preprocess_df[target] = data[target].values

    preprocess_df.to_csv(path, index=False)

    print("SELESAI EUYYY")

    return preprocess_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path ke file input csv mentah")
    parser.add_argument("--output_path", type=str, required=True, help="Path ke csv telah di proses")

    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    preprocess_data(df, args.output_path)