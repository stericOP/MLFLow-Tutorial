import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load wine dataset
wine = load_wine()
x = wine.data
y = wine.target

# Train Test split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state= 42, test_size=.2)

# Params for RF model
max_depth = 5
n_estimators = 10

# Setting up Experiment
mlflow.set_experiment("mlfow-test-experiment")
# RF model 
# with mlflow.start_run(experiment_id=)
with mlflow.start_run():
  rf = RandomForestClassifier(max_depth= max_depth, n_estimators= n_estimators, random_state= 42)
  rf.fit(x_train, y_train)
  y_pred = rf.predict(x_test)

  accuracy = accuracy_score(y_test,y_pred)
  
  mlflow.log_metric("Accuracy", accuracy)
  mlflow.log_param("Max_Depth", max_depth)
  mlflow.log_param("n_estimators", n_estimators)

  # print("Accuracy", accuracy)

  # Confusion Matrix
  cm = confusion_matrix(y_test,y_pred)
  plt.figure(figsize=(6,6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Confusion Matrix")

  # Save figure
  plt.savefig("Confusion-Matrix.png")

  # Load Artifacts
  mlflow.log_artifact("Confusion-Matrix.png")
  mlflow.log_artifact(__file__)

  # Adding Tags 
  mlflow.set_tags({"Author": "Hari",
                   "Project": "Wine Classification"})
  
  # Log the Model 
  mlflow.sklearn.log_model(rf,"Random Forest Model")
