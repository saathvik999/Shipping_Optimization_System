import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/shipping_model.pkl")

factories = [
    "Lot's O' Nuts",
    "Wicked Choccy's",
    "Sugar Shack",
    "Secret Factory",
    "The Other Factory"
]

def simulate_factories(input_df):

    results = []

    for factory in factories:

        predicted = model.predict(input_df)[0]

        results.append({
            "Factory":factory,
            "Predicted Lead Time":predicted
        })

    results = sorted(results,key=lambda x:x["Predicted Lead Time"])

    return results


def recommend_factory(results):

    return results[0]