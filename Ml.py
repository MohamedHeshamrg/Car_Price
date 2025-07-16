# Final model Xgb  With Ordinal Encoder

import pandas as pd
import numpy as np
import timeit
import pickle
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from category_encoders import BinaryEncoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


# Load and preprocess data
base_url = 'H:/Final project/Data/Data cleaned/'
df = pd.concat([pd.read_csv(base_url + f"part{i}.csv") for i in range(1, 7)], ignore_index=True)

df['motor_mi'] = np.log1p(df['motor_mi'])
df.drop(['seller','saledate','market_advantage','sell_month_name','sell_day_name',
         'sell_hour','trim', 'season','sellingprice'], axis=1, inplace=True)

df['condition'] = pd.cut(df['condition'], bins=[0, 10, 20, 30, 40, 50], labels=[1, 2, 3, 4, 5]).astype(int)

x = df.drop('mmr', axis=1)
y = df['mmr']

# Column setup
Ordinal_cols = ['brand', 'model', 'body', 'state', 'color', 'interior', 'time_period']
binary_cols = ['transmission']
scale_cols = ['motor_mi', 'condition', 'model_year','sell_year', 'sell_month', 'sell_day']

# Transformer
preprocessor = ColumnTransformer(transformers=[
    ('Ordinal_enc', Pipeline([
        ('Ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ('scale', RobustScaler())
    ]), Ordinal_cols),
    ('binary_enc', BinaryEncoder(), binary_cols),
    ('num_scaler', RobustScaler(), scale_cols)
])



pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        reg_alpha=0.5,
        reg_lambda=1.0

    ))
])


pipeline=pipeline.fit(x, np.log1p(y))
joblib.dump(pipeline, 'xgb_model.joblib')



print("✅ Model saved cleanly.")