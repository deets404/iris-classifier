from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load the built-in iris dataset (150 flowers, 4 measurements each, 3 species)
iris = load_iris()
X, y = iris.data, iris.target

# Split into training data (80%) and test data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model (a solid, easy-to-use classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check how good it is
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file
joblib.dump(model, 'iris_model.pkl')
print("Model saved as iris_model.pkl")