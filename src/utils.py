import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def evaluate_model(y_true, y_pred, labels=None):
    print(classification_report(y_true, y_pred, target_names=labels))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

def save_model(model, file_path):
    joblib.dump(model, file_path)

def load_model(file_path):
    return joblib.load(file_path)
