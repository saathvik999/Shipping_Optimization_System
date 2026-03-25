import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("/Users/saathvik2005/Documents/UnifiedM/PR2/data/nassau_dataset.csv")

# Convert dates
df["Order Date"] = pd.to_datetime(df["Order Date"],dayfirst=True)
df["Ship Date"] = pd.to_datetime(df["Ship Date"],dayfirst=True)

# Create Lead Time
df["Lead_Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

df = df.dropna()

# Encode categorical variables
encoders = {}

for col in ["Ship Mode","Region","Division","Product Name"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features
X = df[[
    "Ship Mode",
    "Region",
    "Division",
    "Product Name",
    "Units",
    "Cost"
]]

y = df["Lead_Time"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)

# Evaluate
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
r2 = r2_score(y_test,pred)

print("RMSE:",rmse)
print("R2:",r2)

# Save model
os.makedirs("models",exist_ok=True)

joblib.dump(model,"models/shipping_model.pkl")
joblib.dump(encoders,"models/encoders.pkl")

print("Model saved successfully")