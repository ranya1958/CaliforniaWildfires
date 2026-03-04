import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier

# Loads the CSV file
FILE = "California_Fire_Perimeters_(all).csv"
df = pd.read_csv(FILE)

# Drops specified columns
cols_to_remove = [
    "C_METHOD", "FIRE_NUM", "UNIT_ID", "IRWINID", "COMPLEX_NAME", "COMPLEX_ID",
    "OBJECTIVE", "INC_NUM", "COMMENTS", "FORENAME", "STATE"
]

# Removes only the columns that exist in the dataset to avoid errors
df_cleaned = df.drop(
    columns=[col for col in cols_to_remove if col in df.columns])

# Converts ALARM_DATE and CONT_DATE to datetime
df_cleaned["ALARM_DATE"] = pd.to_datetime(df_cleaned["ALARM_DATE"],
                                          errors='coerce')
df_cleaned["CONT_DATE"] = pd.to_datetime(df_cleaned["CONT_DATE"],
                                         errors='coerce')

# Calculates duration and create a new column
df_cleaned["DURATION"] = (df_cleaned["CONT_DATE"] -
                          df_cleaned["ALARM_DATE"]).dt.days

# Drops original date columns
df_cleaned = df_cleaned.drop(columns=["ALARM_DATE", "CONT_DATE"])
df_cleaned = df_cleaned.dropna(subset=["DURATION", "YEAR_", "Shape__Area", "Shape__Length", "CAUSE"])

# Displays columns of cleaned data
print(list(df_cleaned.columns))
print()

# Displays cleaned data
print(df_cleaned.head(20))

# Algorithms
# 1. KNN Classfier
# X = df_cleaned[["DURATION", "YEAR_", "Shape__Area", "Shape__Length"]]
X = df_cleaned[["DURATION"]]
y = df_cleaned["CAUSE"]
# Splits data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# features: year, size, mayb smthn else duration
# target: cause
def knn(X_train, X_test, y_train, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predicted = knn.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)
    return predicted, accuracy


pred, acc = knn(X_train, X_test, y_train, y_test, 15)
print(f"KNN classifier accuracy: {acc}")

# 2. Multiple Linear Regression
X = df_cleaned[["DURATION", "YEAR_", "Shape__Length"]]
y = df_cleaned["Shape__Area"]
# Splits data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def linear_r(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    predicted = lr.predict(X_test)

    mse = mean_squared_error(y_test, predicted)
    print(f"Mean Squared Error: {mse}")
    return predicted, mse


pred, mse = linear_r(X_train, X_test, y_train, y_test)
print(f"Multiple Linear Regression accuracy: {mse}")

# 3. Random Forest Classifier
X = df_cleaned[["DURATION", "YEAR_", "Shape__Area", "Shape__Length"]]
y = df_cleaned["CAUSE"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def random_forest_classifier(X_train, X_test, y_train, y_test, n_estimators=100):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    rf.fit(X_train, y_train)
    predicted = rf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)
    f1 = f1_score(y_test, predicted, average='weighted')
    print(f"Random Forest Classifier accuracy: {accuracy}")
    print(f"Random Forest F1 score: {f1}")
    return predicted, accuracy, f1

rf_pred, rf_acc, rf_f1 = random_forest_classifier(X_train, X_test, y_train, y_test)


# Visualizations
# Correlation matrix - Relationship between frequency and average duration of fires over the years
def correlation_analysis(df):
    yearly = df.groupby("YEAR_").agg({
        "DURATION": "mean",
        "FIRE_NAME": "count"
    }).rename(columns={
        "FIRE_NAME": "FREQUENCY",
        "DURATION": "AVG_DURATION"
    })

    yearly["YEAR_"] = yearly.index
    corr_matrix = yearly[["YEAR_", "FREQUENCY", "AVG_DURATION"]].corr()

    print("\nCorrelation Matrix:")
    print(corr_matrix)
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.show()

    return corr_matrix

correlation_analysis(df_cleaned)
