import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Charger dataset
data = pd.read_csv("dataset.csv")

# Séparer features / label
X = data.drop(columns=["label"])
y = data["label"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Modèle (simple et efficace)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("Précision test:", model.score(X_test, y_test))

# Sauvegarde
joblib.dump(model, "exercise_model.pkl")
print("Modèle sauvegardé -> exercise_model.pkl")
