import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from google.colab import drive
drive.mount('/content/drive')
data = pd.read_csv("/content/drive/MyDrive/ml2/Disease_train.csv") #loading Disease_train
data.isnull().sum()
data.isnull().sum().sum()
X = data.iloc[:, :-2]  # Features: all columns except the last one and last but one
y = data.iloc[:,-1]   # Target: the last column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #seprating the data for training and testing
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
ans=clf.predict(X_test)
accuracy = accuracy_score(y_test, ans)
print(f"Accuracy: {accuracy*100:.2f}%") #checking the accuracy of the model
data1 = pd.read_csv("/content/drive/MyDrive/dataset/ml-2/Disease_test.csv")   #loading Disease_test
 # Extract patient_id from the last column
patient_id = data1.iloc[:, -1]
# features: all columns except the last one

X2 = data1.iloc[:, :-1] 
predictions = clf.predict(X2)
print(predictions)
results = pd.DataFrame({
    'patient_id': patient_id,
    'prediction': predictions
})

# Print the DataFrame
print(results)
results.to_csv("/content/drive/MyDrive/dataset/ml-2/Disease_predictions.csv", index=False)