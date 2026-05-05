import pandas as pd
import numpy as np
import urllib.request
import re
import json
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Expanded city list — slug must match InsideAirbnb URL format
CITIES = [
    'austin', 'dallas', 'los-angeles', 'new-york-city', 'chicago',
    'seattle', 'san-francisco', 'boston', 'denver', 'miami',
    'nashville', 'portland', 'washington-dc', 'san-diego', 'new-orleans'
]

def get_city_urls(city):
    print(f"Fetching data URLs for {city}...")
    url = 'http://insideairbnb.com/get-the-data/'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            links = re.findall(r'href=[\'\"]([^\'\"]+' + city + r'[^\'\"]+listings\.csv\.gz)', html)
            return links
    except Exception as e:
        print(f'Error fetching URL for {city}: {e}')
        return []

def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    return float(str(price_str).replace('$', '').replace(',', ''))

def clean_baths(bath_str):
    if pd.isna(bath_str):
        return 1.0
    try:
        num = re.findall(r'\d+\.?\d*', str(bath_str))
        if num:
            return float(num[0])
        return 1.0
    except:
        return 1.0

def count_amenities(amenities_str):
    """Count amenities from JSON-like list string."""
    if pd.isna(amenities_str) or not str(amenities_str).strip():
        return 0
    try:
        items = json.loads(str(amenities_str))
        return len(items)
    except:
        # Fallback: count comma-separated items
        return len(str(amenities_str).split(','))

def extract_premium_features(df):
    """
    Regex-based feature extraction — matches exactly what is used at inference
    time in the app so training labels and inference flags are consistent.
    """
    desc = df['description'].fillna('').str.lower()
    df['has_pool'] = desc.str.contains(r'\bpool\b', regex=True).astype(int)
    df['has_hot_tub'] = desc.str.contains('hot tub|jacuzzi', regex=True).astype(int)
    df['has_view'] = desc.str.contains(r'\bview\b', regex=True).astype(int)
    df['is_luxury'] = desc.str.contains('luxury|premium|high-end', regex=True).astype(int)
    return df

def train_model():
    all_dfs = []

    # Columns to pull — extra columns for new features, will be dropped if missing
    desired_cols = [
        'price', 'bedrooms', 'accommodates', 'neighbourhood_cleansed',
        'room_type', 'bathrooms_text', 'description',
        'review_scores_rating', 'number_of_reviews',
        'host_is_superhost', 'instant_bookable', 'amenities'
    ]

    for city in CITIES:
        urls = get_city_urls(city)
        if not urls:
            print(f"Skipping {city}, no URLs found.")
            continue

        for url in urls:
            print(f"Attempting download for {city} from {url}...")
            try:
                # Only request columns that exist in the CSV
                df_peek = pd.read_csv(url, compression='gzip', nrows=1, low_memory=False)
                available = [c for c in desired_cols if c in df_peek.columns]
                df = pd.read_csv(url, compression='gzip', usecols=available, low_memory=False)

                df['price'] = df['price'].apply(clean_price)
                if df['price'].isnull().all():
                    print(f"--> Warning: 100% missing prices. Trying older dataset...")
                    continue

                formatted_city = city.replace('-', ' ').title()
                df['city'] = formatted_city

                year_match = re.search(r'/(\d{4})-\d{2}-\d{2}/', url)
                df['data_year'] = year_match.group(1) if year_match else '2025'

                all_dfs.append(df)
                print(f"--> Success! Loaded {len(df)} listings for {formatted_city}.")
                break

            except Exception as e:
                print(f"--> Failed to load: {e}")

    if not all_dfs:
        print("No data loaded. Exiting.")
        return

    print("\nMerging datasets...")
    df = pd.concat(all_dfs, ignore_index=True)

    print("Cleaning data...")
    df = df.dropna(subset=['price', 'bedrooms', 'accommodates', 'neighbourhood_cleansed', 'room_type', 'city'])

    # Filter extreme outliers — raised upper cap to 2000 to include legitimate luxury listings
    df = df[(df['price'] > 20) & (df['price'] < 2000)]
    df = df[df['bedrooms'] <= 10]

    df['baths'] = df['bathrooms_text'].apply(clean_baths)
    df['accommodates'] = df['accommodates'].fillna(1).astype(float)
    df['bedrooms'] = df['bedrooms'].astype(float)
    df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].astype(str)

    # --- New features (with safe fallbacks if column missing) ---
    if 'review_scores_rating' in df.columns:
        df['review_score'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())
    else:
        df['review_score'] = 4.5  # neutral fallback

    if 'number_of_reviews' in df.columns:
        df['num_reviews'] = df['number_of_reviews'].fillna(0).clip(upper=500)
    else:
        df['num_reviews'] = 0

    if 'host_is_superhost' in df.columns:
        df['is_superhost'] = (df['host_is_superhost'] == 't').astype(int)
    else:
        df['is_superhost'] = 0

    if 'instant_bookable' in df.columns:
        df['instant_bookable'] = (df['instant_bookable'] == 't').astype(int)
    else:
        df['instant_bookable'] = 0

    if 'amenities' in df.columns:
        df['amenity_count'] = df['amenities'].apply(count_amenities).clip(upper=100)
    else:
        df['amenity_count'] = 0

    # Text premium features — same regex used at inference (fixes training/inference mismatch)
    df = extract_premium_features(df)

    # Save neighborhood lookup per city
    city_neighborhoods = {}
    for c in df['city'].unique():
        neighborhoods = sorted(df[df['city'] == c]['neighbourhood_cleansed'].unique().tolist())
        city_neighborhoods[c] = neighborhoods
    joblib.dump(city_neighborhoods, 'city_neighborhoods.pkl')

    # Save neighborhood price stats for comps display in app
    neighborhood_stats = (
        df.groupby(['city', 'neighbourhood_cleansed'])['price']
        .agg(median='median', p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    joblib.dump(neighborhood_stats, 'neighborhood_stats.pkl')

    print(f"Total Clean National Listings: {len(df)}")

    # --- Feature sets ---
    categorical_features = ['city', 'neighbourhood_cleansed', 'room_type', 'data_year']
    numeric_features = [
        'bedrooms', 'baths', 'accommodates',
        'review_score', 'num_reviews', 'is_superhost', 'instant_bookable', 'amenity_count',
        'has_pool', 'has_hot_tub', 'has_view', 'is_luxury'
    ]

    X = df[categorical_features + numeric_features]
    y = df['price']

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
        remainder='passthrough'
    )

    print("Executing 3-Fold CV with Log-Transform and Early Stopping...")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmse_scores = []

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_trans = preprocessor.fit_transform(X_train)
        X_val_trans = preprocessor.transform(X_val)

        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)

        # max_depth kept at 6 to limit overfitting given expanded feature set
        model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,  # added sampling to reduce overfitting
            random_state=42, early_stopping_rounds=20
        )
        model.fit(X_train_trans, y_train_log, eval_set=[(X_val_trans, y_val_log)], verbose=False)

        preds_log = model.predict(X_val_trans)
        preds = np.expm1(preds_log)

        score = root_mean_squared_error(y_val, preds)
        rmse_scores.append(score)
        print(f"Fold {i+1} RMSE: ${score:.2f} (Stopped at tree {model.best_iteration})")

    print(f"===================================")
    print(f"Average National CV RMSE: ${np.mean(rmse_scores):.2f}")
    print(f"===================================")

    print("Training final master model...")
    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(X, y, test_size=0.1, random_state=42)

    X_train_trans_f = preprocessor.fit_transform(X_train_f)
    X_val_trans_f = preprocessor.transform(X_val_f)

    y_train_log_f = np.log1p(y_train_f)
    y_val_log_f = np.log1p(y_val_f)

    final_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=20
    )
    final_model.fit(X_train_trans_f, y_train_log_f, eval_set=[(X_val_trans_f, y_val_log_f)], verbose=False)

    # --- Feature Importances ---
    print("Extracting Feature Importances...")
    feature_names = preprocessor.get_feature_names_out()
    importances = final_model.feature_importances_

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)

    def clean_feature_name(name):
        name = name.replace('cat__', '').replace('remainder__', '')
        if 'room_type_' in name: return name.replace('room_type_', 'Room Type: ')
        if 'city_' in name: return name.replace('city_', 'City: ')
        if 'neighbourhood_cleansed_' in name: return name.replace('neighbourhood_cleansed_', 'Neighborhood: ')
        if 'data_year_' in name: return name.replace('data_year_', 'Year: ')
        return name.replace('_', ' ').title()

    importance_df['Feature'] = importance_df['Feature'].apply(clean_feature_name)
    joblib.dump(importance_df, 'feature_importances.pkl')

    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(final_model, 'airbnb_model.pkl')

    # Save feature name list so SHAP values can be labeled in the app
    joblib.dump(list(feature_names), 'feature_names.pkl')

    print("Pipeline saved successfully!")

if __name__ == "__main__":
    train_model()
