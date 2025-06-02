# Script to train machine learning model.
import pandas as pd
import joblib
import os
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split

# Add code to load in the data.
data = pd.read_csv("data/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
# Split the data into training and testing sets.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, 
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Evaluate the model
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Model performance on test set:\nPrecision: {precision:.3f}  Recall: {recall:.3f}  F1: {fbeta:.3f}")

# Save model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/rf_model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/lb.pkl")

# Function to log performance on slices of multiple categorical features
def evaluate_slices(test_data, categorical_features, model, encoder, lb):
    with open("model/slice_output.txt", "w") as f:
        for feature in categorical_features:
            for value in test_data[feature].unique():
                slice_data = test_data[test_data[feature] == value]
                if slice_data.shape[0] == 0:
                    continue  # skip if slice is empty
                X_slice, y_slice, _, _ = process_data(
                    slice_data,
                    categorical_features=categorical_features,
                    label="salary",
                    training=False,
                    encoder=encoder,
                    lb=lb
                )
                preds = inference(model, X_slice)
                precision, recall, fbeta = compute_model_metrics(y_slice, preds)
                f.write(
                    f"{feature} = {value} --> precision: {precision:.3f}, recall: {recall:.3f}, f1: {fbeta:.3f}\n"
                )

# Evaluate model performance on all categorical slices
evaluate_slices(test, cat_features, model, encoder, lb)