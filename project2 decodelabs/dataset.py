import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------------------
# Step 1: Load and Understand the Dataset
# -------------------------------------------------------------
# Load a classic classification dataset (Iris flowers)
iris = load_iris()

# Create a DataFrame for easy inspection
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("--- Dataset Head (First 5 Rows) ---")
print(df.head())

print("\n--- Class Names ---")
for idx, name in enumerate(iris.target_names):
    print(f"Class {idx}: {name}")

# Separate features (X) and target label (y)
X = iris.data
y = iris.target

# -------------------------------------------------------------
# Step 2: Split Data into Training and Testing Sets
# -------------------------------------------------------------
# 80% of the data is used for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")

# -------------------------------------------------------------
# Step 3: Apply a Classification Algorithm (Decision Tree)
# -------------------------------------------------------------
# Initialize the model
model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# -------------------------------------------------------------
# Step 4: Evaluate the Model
# -------------------------------------------------------------
# Predict on unseen test data
y_pred = model.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n--- Model Accuracy: {acc * 100:.2f}% ---")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------------------
# Step 5: Test with Custom/New Data
# -------------------------------------------------------------
# Example sample: [sepal length, sepal width, petal length, petal width]
new_sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_sample)
predicted_class = iris.target_names[prediction[0]]

print(f"\nPrediction for new sample {new_sample}: {predicted_class}")