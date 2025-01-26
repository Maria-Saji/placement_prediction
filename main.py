# %% [md]
"""
## Placement Prediction (Classification Problem)
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# %%
raw_data = pd.read_csv("data/placementdata.csv")
print(raw_data.head())
print(raw_data.columns)
print(raw_data.shape)

# %%
print(raw_data.info())

# %% [md]
"""
There are 10k all non-null values, nice!
"""

# %%
print(raw_data.describe())

# %%
# Some renaming/conversion
processed_data = raw_data.copy()
processed_data["ExtracurricularActivities"] = processed_data["ExtracurricularActivities"].apply(lambda x: True if x == "Yes" else False)
processed_data["PlacementTraining"] = processed_data["PlacementTraining"].apply(lambda x: True if x == "Yes" else False)
processed_data["PlacementStatus"] = processed_data["PlacementStatus"].apply(lambda x: True if x == "Placed" else False)

print(processed_data.head())


# %% [md]
"""
There are no null-values, no outliers, no any -ves. This is very clean data!
SSC_Marks lies in 55 to 90, while HSC_Marks ranges 57 to 88 with 69.15 and 74.5 mean values respectively
CGPA is from 6.5 to 9.1 and a 7.69 mean.
Let's take a look at some graphs
"""

# %%
processed_data[["CGPA", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]].hist(bins=60, figsize=(12, 12))
plt.show()

# %% [md]
"""
Absolutely no data cleaning is required
"""

# %%
placement_data_filtered = processed_data[
    [
        "CGPA",
        "Internships",
        "Projects",
        "Workshops/Certifications",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "ExtracurricularActivities",
        "PlacementTraining",
        "SSC_Marks",
        "HSC_Marks",
        "PlacementStatus",
    ]
]

print(placement_data_filtered.dtypes)

plt.title("Extra Curricular Activities")
print(placement_data_filtered["ExtracurricularActivities"].value_counts().plot(kind="pie"))
plt.show()

plt.title("Placement Training")
print(placement_data_filtered["PlacementTraining"].value_counts().plot(kind="pie"))
plt.show()

plt.title("Placement Status")
print(placement_data_filtered["PlacementStatus"].value_counts().plot(kind="pie"))
plt.show()

# %%
categorical_columns = [
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "ExtracurricularActivities",
    "PlacementTraining",
    "PlacementStatus",
]

numerical_columns = ["CGPA", "AptitudeTestScore", "SoftSkillsRating", "SSC_Marks", "HSC_Marks"]
all_columns = categorical_columns + numerical_columns

# %%
main_placement_df = pd.DataFrame(placement_data_filtered[all_columns])
print(main_placement_df.shape)
print(main_placement_df.head())

# %% [md]
"""
## Benchmark Model
"""

# %%
data_benchmark = main_placement_df.copy()

input_features_benchmark = [
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "ExtracurricularActivities",
    "PlacementTraining",
    "CGPA",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "SSC_Marks",
    "HSC_Marks",
]

target_feature_benchmark = "PlacementStatus"

# %% [md]
"""
## Train Test Split
"""

# %%
X_bench = data_benchmark[input_features_benchmark]
y_bench = data_benchmark[target_feature_benchmark]

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bench, y_bench, test_size=0.2, random_state=42)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_b, y_train_b)

# %%
y_pred_b = clf.predict(X_test_b)

print("Benchmark Report:\n", classification_report(y_pred=y_pred_b, y_true=y_test_b))

# %%
def accuracy(y_pred, y_true):
    return round(np.sum(y_pred == y_true) / len(y_pred), 5)

# %%
# RandomForest

X, y = X_bench, y_bench
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = RandomForestClassifier(max_depth=10)
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)
print("Benchmark Report:\n", classification_report(y_pred=y_pred, y_true=y_test))
print(f"Accuracy: {accuracy(y_pred, y_test) * 100}%")

# %%
# GradientBoosting

X, y = X_bench, y_bench
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)
print("Benchmark Report:\n", classification_report(y_pred=y_pred, y_true=y_test))
print(f"Accuracy: {accuracy(y_pred, y_test) * 100}%")

# %%
# DecisionTree

X, y = X_bench, y_bench
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)
print("Benchmark Report:\n", classification_report(y_pred=y_pred, y_true=y_test))
print(f"Accuracy: {accuracy(y_pred, y_test) * 100}%")

# %%
# MLPClassifier

X, y = X_bench, y_bench
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = MLPClassifier()
clf.fit(X_train, y_train)

# %%
y_pred = clf.predict(X_test)
print("Benchmark Report:\n", classification_report(y_pred=y_pred, y_true=y_test))
print(f"Accuracy: {accuracy(y_pred, y_test) * 100}%")
# %%
# GridSearchCV

X, y = X_bench, y_bench
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    cv=5
)
clf.fit(X_train, y_train)
# %%
model = clf.best_estimator_
print(model)

# %%
y_pred = clf.predict(X_test)
print("Benchmark Report:\n", classification_report(y_pred=y_pred, y_true=y_test))
print(f"Accuracy: {accuracy(y_pred, y_test) * 100}%")
# %%
import pickle

with open("random_forest_clf.pkl", 'wb') as fp:
    pickle.dump(model, fp)

# %% [md]
"""
Now let's try Neural Networks
"""

# %%
import keras
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %%
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(10,)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# %%
model.summary()

# %%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
history = model.fit(X_train, y_train, epochs=50, batch_size=32)

# %%
pd.DataFrame(history.history).plot(figsize=(15, 8))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# %%
model.evaluate(X_test, y_test)

# %%
y_prob = model.predict(X_test)
y_prob

# %%
model.save("placement_model.keras")
