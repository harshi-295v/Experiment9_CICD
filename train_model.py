# Name: GLV Harshitha
# Registration Number: 231FA04F53

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, "model.joblib")

print("Model training completed successfully.")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Model accuracy:", round(accuracy, 4))
print("Model saved as model.joblib")