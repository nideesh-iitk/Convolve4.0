#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation


# In[5]:


train_df_raw=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')


# In[6]:


print("Train dataset shape:", train_df_raw.shape)
print("Test dataset shape:", test_df.shape)

print("\nTrain dataset preview:")
display(train_df_raw.head())

print("\nTest dataset preview:")
display(test_df.head())

print("\nMissing values in train dataset:")
print(train_df_raw.isnull().sum())

print("\nMissing values in test dataset:")
print(test_df.isnull().sum())


# In[ ]:


# Now we have got to know the basic things about the dataset 
# and have also checked the presence of missing values, 
# we will handle it in the upcoming blocks


# In[7]:


train_df = train_df_raw.sort_values(["date_id", "time_id"]).reset_index(drop=True)
test_df  = test_df.sort_values(["date_id", "time_id"]).reset_index(drop=True)


# In[8]:


for df in [train_df, test_df]:
    df["f1_missing"] = df["f1"].isna().astype(int)

train_df["f1"] = train_df.groupby("symbol_id")["f1"].ffill()
test_df["f1"]  = test_df.groupby("symbol_id")["f1"].ffill()

train_df["f1"] = train_df["f1"].fillna(0)
test_df["f1"]  = test_df["f1"].fillna(0)


# In[9]:


exclude_cols = ["Id", "date_id", "time_id", "symbol_id", "y"]

feature_cols = sorted(set(
    c for c in train_df.columns
    if c.startswith("f") and c not in exclude_cols
))

print("Number of features:", len(feature_cols))


# In[10]:


last_date = train_df["date_id"].max()

train_cv = train_df[train_df["date_id"] < last_date]
val_cv   = train_df[train_df["date_id"] == last_date]

X_train = train_cv[feature_cols]
y_train = train_cv["y"]

X_val = val_cv[feature_cols]
y_val = val_cv["y"]


# In[11]:


lgb_train = lgb.Dataset(X_train, y_train)
lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42
}

model_full = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "val"],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100)
    ]
)



# In[ ]:


from sklearn.metrics import mean_squared_error

val_preds_full = model_full.predict(
    X_val, 
    num_iteration=model_full.best_iteration
)

rmse_full = mean_squared_error(y_val, val_preds_full, squared=False)

print("RMSE (Impute + Missing Indicator):", rmse_full)


# In[ ]:


X_test = test_df[feature_cols]

test_preds_full = model_full.predict(
    X_test,
    num_iteration=model_full.best_iteration
)

submission_full = pd.DataFrame({
    "Id": test_df["Id"],
    "y": test_preds_full
})

submission_full.to_csv("submission_full.csv", index=False)
print("Submission CSV saved: submission_full.csv")


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

# Get importance from LightGBM
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model_full.feature_importance()
}).sort_values(by="importance", ascending=False)

# Quick table
print(importance)

# Optional plot
plt.figure(figsize=(10,6))
plt.barh(importance["feature"], importance["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("LightGBM Feature Importance")
plt.show()


# In[ ]:


# We get the top features, particularly, f0 is a dominant feature
# Let us train with the top features


# In[ ]:


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

top_features = [
    "f0", "f15", "f20", "f18", "f25", "f10", "f16", "f9", "f3",
    "f21", "f23", "f13", "f17", "f8", "f7"
]

print("Top features used for retrain:", top_features)

X = train_df[top_features]
y = train_df["y"]

# 80/20 random split (can switch to time-based later)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

model_top = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)


val_preds = model_top.predict(X_val, num_iteration=model_top.best_iteration)
rmse_val = root_mean_squared_error(y_val, val_preds)
print("Validation RMSE (Top Features):", rmse_val)

X_test = test_df[top_features]
test_preds = model_top.predict(X_test, num_iteration=model_top.best_iteration)

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "y": test_preds
})

submission.to_csv("submission_top_features.csv", index=False)
print("Submission CSV saved: submission_top_features.csv")


# In[ ]:


# Sort dataset by (date_id, time_id) to respect temporal order
train_df_sorted = train_df.sort_values(by=["date_id", "time_id"]).reset_index(drop=True)


# In[ ]:


# Compute split index
split_idx = int(len(train_df_sorted) * 0.8)

# Train = first 80%, Validation = last 20%
X_train = train_df_sorted.loc[:split_idx-1, top_features]
y_train = train_df_sorted.loc[:split_idx-1, "y"]

X_val   = train_df_sorted.loc[split_idx:, top_features]
y_val   = train_df_sorted.loc[split_idx:, "y"]

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)


# In[ ]:


import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)


# In[ ]:


params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

model_time = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)


# In[ ]:


from sklearn.metrics import root_mean_squared_error

val_preds = model_time.predict(X_val, num_iteration=model_time.best_iteration)
rmse_val = root_mean_squared_error(y_val, val_preds)
print("Time-based Validation RMSE (Top Features):", rmse_val)


# In[ ]:


import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

train_df_sorted = train_df.sort_values(by=["date_id", "time_id"]).reset_index(drop=True)

split_idx = int(len(train_df_sorted) * 0.8)  # 80% train, 20% validation

top_features = [
    "f0", "f15", "f20", "f18", "f25", "f10", "f16", "f9", "f3",
    "f21", "f23", "f13", "f17", "f8", "f7"
]

X_train = train_df_sorted.loc[:split_idx-1, top_features]
y_train = train_df_sorted.loc[:split_idx-1, "y"]

X_val = train_df_sorted.loc[split_idx:, top_features]
y_val = train_df_sorted.loc[split_idx:, "y"]

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

model_time = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

val_preds = model_time.predict(X_val, num_iteration=model_time.best_iteration)
rmse_val = root_mean_squared_error(y_val, val_preds)
print("Time-based Validation RMSE (Top Features):", rmse_val)

X_test = test_df[top_features]
test_preds = model_time.predict(X_test, num_iteration=model_time.best_iteration)

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "y": test_preds
})

submission.to_csv("submission_time_top_features.csv", index=False)
print("Submission CSV saved: submission_time_top_features.csv")


# In[ ]:


get_ipython().system('pip install optuna')


# In[ ]:


# We try hyperparameter optimisation to tune the model better


# In[ ]:


import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

def objective(trial):
    # Sample hyperparameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "seed": 42
    }

    # Create LightGBM datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # Train model with early stopping
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # Predict on validation
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y_val, val_preds, squared=False)
    
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)  # start with 30 trials

print("Best RMSE:", study.best_value)
print("Best params:", study.best_params)

best_params = study.best_params
best_params["objective"] = "regression"
best_params["metric"] = "rmse"
best_params["seed"] = 42

lgb_train_full = lgb.Dataset(X_train, y_train)
lgb_val_full   = lgb.Dataset(X_val, y_val, reference=lgb_train_full)

model_opt = lgb.train(
    best_params,
    lgb_train_full,
    num_boost_round=1000,
    valid_sets=[lgb_train_full, lgb_val_full],
    valid_names=["train", "val"],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

val_preds_opt = model_opt.predict(X_val, num_iteration=model_opt.best_iteration)
rmse_val_opt = mean_squared_error(y_val, val_preds_opt, squared=False)
print("Validation RMSE after Optuna tuning:", rmse_val_opt)

X_test_top = test_df[top_features]
test_preds_opt = model_opt.predict(X_test_top, num_iteration=model_opt.best_iteration)

submission_opt = pd.DataFrame({
    "Id": test_df["Id"],
    "y": test_preds_opt
})
submission_opt.to_csv("submission_optuna.csv", index=False)
print("Submission CSV saved: submission_optuna.csv")


# In[ ]:


# Trying lag and rolling features


# In[ ]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna

top_features = [
    "f0", "f15", "f20", "f18", "f25", "f10", "f16", "f9", "f3",
    "f21", "f23", "f13", "f17", "f8", "f7"
]

lag_cols = ["f0", "f15"]
roll_cols = ["f0", "f15"]
lag_n = 1
roll_n = 3

train_df_sorted = train_df.sort_values(["symbol_id", "date_id", "time_id"]).reset_index(drop=True)

for col in lag_cols:
    train_df_sorted[f"{col}_lag{lag_n}"] = train_df_sorted.groupby("symbol_id")[col].shift(lag_n)

for col in roll_cols:
    train_df_sorted[f"{col}_roll{roll_n}"] = train_df_sorted.groupby("symbol_id")[col].shift(1).rolling(roll_n, min_periods=1).mean()

# Drop any rows with NaNs created by lag/roll
train_df_sorted = train_df_sorted.dropna().reset_index(drop=True)

# Update feature list
all_features = top_features + [f"{col}_lag{lag_n}" for col in lag_cols] + [f"{col}_roll{roll_n}" for col in roll_cols]

split_idx = int(len(train_df_sorted) * 0.8)
X_train = train_df_sorted.loc[:split_idx-1, all_features]
y_train = train_df_sorted.loc[:split_idx-1, "y"]

X_val = train_df_sorted.loc[split_idx:, all_features]
y_val = train_df_sorted.loc[split_idx:, "y"]

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "seed": 42
    }

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y_val, val_preds, squared=False)
    return rmse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best RMSE:", study.best_value)
print("Best params:", study.best_params)

best_params = study.best_params
best_params.update({"objective": "regression", "metric": "rmse", "seed": 42})

lgb_train_full = lgb.Dataset(X_train, y_train)
lgb_val_full   = lgb.Dataset(X_val, y_val, reference=lgb_train_full)

model_final = lgb.train(
    best_params,
    lgb_train_full,
    num_boost_round=1000,
    valid_sets=[lgb_train_full, lgb_val_full],
    valid_names=["train", "val"],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

val_preds_final = model_final.predict(X_val, num_iteration=model_final.best_iteration)
rmse_final = mean_squared_error(y_val, val_preds_final, squared=False)
print("Final Validation RMSE:", rmse_final)

# For test set, create lag/rolling features in the same way
test_df_sorted = test_df.sort_values(["symbol_id", "date_id", "time_id"]).reset_index(drop=True)

for col in lag_cols:
    test_df_sorted[f"{col}_lag{lag_n}"] = test_df_sorted.groupby("symbol_id")[col].shift(lag_n)

for col in roll_cols:
    test_df_sorted[f"{col}_roll{roll_n}"] = test_df_sorted.groupby("symbol_id")[col].shift(1).rolling(roll_n, min_periods=1).mean()

# For simplicity, fill NaNs in test with train mean (or 0)
for col in [f"{c}_lag{lag_n}" for c in lag_cols] + [f"{c}_roll{roll_n}" for c in roll_cols]:
    test_df_sorted[col] = test_df_sorted[col].fillna(train_df_sorted[col].mean())

X_test_final = test_df_sorted[all_features]
test_preds_final = model_final.predict(X_test_final, num_iteration=model_final.best_iteration)

submission_final = pd.DataFrame({
    "Id": test_df_sorted["Id"],
    "y": test_preds_final
})

submission_final.to_csv("submission_top_lag_optuna.csv", index=False)
print("Submission CSV saved: submission_top_lag_optuna.csv")


# In[ ]:


get_ipython().system('pip install shap')


# In[ ]:


# Doing SHAP to further check explainability


# In[ ]:


import shap
import numpy as np

# Sample from training data used for this model
sample_size = 20000
X_shap = X_train.sample(sample_size, random_state=42)

print("SHAP sample shape:", X_shap.shape)


# explainer = shap.TreeExplainer(model_final)  # lag/rolling model
# shap_values = explainer.shap_values(X_shap)
# 
# print("SHAP values computed")
# 

# In[ ]:


explainer = shap.TreeExplainer(model_final)  # lag/rolling model
shap_values = explainer.shap_values(X_shap)

print("SHAP values computed")


# In[ ]:


shap.summary_plot(
    shap_values,
    X_shap,
    plot_type="bar",
    max_display=20
)


# In[ ]:


shap.summary_plot(
    shap_values,
    X_shap,
    max_display=15
)


# In[ ]:


# Experimenting by removing the most dominant feature f0


# In[ ]:


features_no_f0 = [f for f in top_features if f != "f0"]

# retrain model with same params
X_train_nf = X_train[features_no_f0]
X_val_nf   = X_val[features_no_f0]

lgb_train_nf = lgb.Dataset(X_train_nf, y_train)
lgb_val_nf   = lgb.Dataset(X_val_nf, y_val)

model_nf = lgb.train(
    best_params,
    lgb_train_nf,
    num_boost_round=1000,
    valid_sets=[lgb_train_nf, lgb_val_nf],
    callbacks=[lgb.early_stopping(50)]
)

val_preds_nf = model_nf.predict(X_val_nf, num_iteration=model_nf.best_iteration)
rmse_nf = root_mean_squared_error(y_val, val_preds_nf)

print("Validation RMSE without f0:", rmse_nf)


# In[ ]:


# Removing f0 alone is insufficient as there is leakage of f0 into the rolling and lag features, so we remove all related features too


# In[ ]:


# Identify all f0-related columns
f0_cols = [c for c in train_df.columns if c.startswith("f0")]

print("Removing f0-related columns:")
print(f0_cols)

# Drop them from train and test
train_no_f0 = train_df.drop(columns=f0_cols)
test_no_f0  = test_df.drop(columns=f0_cols)


# In[ ]:


exclude_cols = ["Id", "date_id", "time_id", "symbol_id", "y"]
feature_cols_no_f0 = [c for c in train_no_f0.columns if c not in exclude_cols]

print("Number of features (no f0):", len(feature_cols_no_f0))


# In[ ]:


# Sort by time
train_no_f0_sorted = train_no_f0.sort_values(
    by=["date_id", "time_id"]
).reset_index(drop=True)

split_idx = int(len(train_no_f0_sorted) * 0.8)

X_train = train_no_f0_sorted.loc[:split_idx-1, feature_cols_no_f0]
y_train = train_no_f0_sorted.loc[:split_idx-1, "y"]

X_val   = train_no_f0_sorted.loc[split_idx:, feature_cols_no_f0]
y_val   = train_no_f0_sorted.loc[split_idx:, "y"]

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)


# In[ ]:


import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

model_no_f0 = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)


# In[ ]:


from sklearn.metrics import root_mean_squared_error

val_preds = model_no_f0.predict(
    X_val, num_iteration=model_no_f0.best_iteration
)

rmse_no_f0 = root_mean_squared_error(y_val, val_preds)
print("Validation RMSE (NO f0 at all):", rmse_no_f0)

