import pandas as pd
import numpy as np

from sklearn.metrics import r2_score
from xgboost import XGBRegressor


# ======================
# LOAD DATA
# ======================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# ======================
# FEATURE ENGINEERING
# ======================
def create_features(df):

    df = pd.get_dummies(df, columns=['option_type'], drop_first=True)

    df['moneyness_squared'] = df['moneyness']**2
    df['moneyness_log'] = np.log(df['moneyness'].clip(1e-6))

    df['tau_squared'] = df['tau']**2
    df['tau_log'] = np.log(df['tau'].clip(1e-6))

    df['interaction_term'] = df['moneyness'] * df['tau']

    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

    return df


train = create_features(train)
test = create_features(test)


# ======================
# REMOVE MISSING TARGET
# ======================
train = train[train['iv_observed'].notnull()].copy()


# ======================
# SORT BY TIME 
# ======================
train = train.sort_values('date')


# ======================
# DATE IV FEATURE (NO LEAKAGE)
# ======================
train['date_iv_mean'] = (
    train.groupby('date')['iv_observed']
    .transform('mean')
)

# shift → avoid leakage
train['date_iv_mean'] = train['date_iv_mean'].shift(1)

# fill missing
global_mean = train['iv_observed'].mean()
train['date_iv_mean'].fillna(global_mean, inplace=True)

# apply to test
date_mean_map = train.groupby('date')['iv_observed'].mean()
test['date_iv_mean'] = test['date'].map(date_mean_map)
test['date_iv_mean'].fillna(global_mean, inplace=True)


# ======================
# FEATURES
# ======================
features = [
    'moneyness',
    'tau',
    'moneyness_squared',
    'moneyness_log',
    'tau_log',
    'interaction_term',
    'option_type_put',
    'day_of_week',
    'week_of_year',
    'date_iv_mean'   # ⭐ MOST IMPORTANT
]


# ======================
# TIME-BASED SPLIT
# ======================
split = int(0.8 * len(train))

train_data = train.iloc[:split]
val_data = train.iloc[split:]


X_train = train_data[features]
y_train = train_data['iv_observed']

X_val = val_data[features]
y_val = val_data['iv_observed']


# ======================
# MODEL 
# ======================
model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)


# ======================
# VALIDATION
# ======================
val_pred = model.predict(X_val)

print("="*40)
print("VALIDATION R2:", r2_score(y_val, val_pred))
print("="*40)


# ======================
# TRAIN ON FULL DATA
# ======================
X_full = train[features]
y_full = train['iv_observed']

model.fit(X_full, y_full)


# ======================
# TEST PREDICTION
# ======================
X_test = test[features].copy()

final_pred = model.predict(X_test)


# ======================
# SUBMISSION
# ======================
test['iv_predicted'] = final_pred

submission = test[['row_id', 'iv_predicted']]
submission.to_csv('submission.csv', index=False)

print("🚀 Submission ready!")