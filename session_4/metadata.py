MODELS_FOLDER = "/Users/andresrodartee/mlops-and-system-design/session_4/models"
DATASETS_FOLDER = "/Users/andresrodartee/mlops-and-system-design/session_4/Class Notes/datasets"
MODEL_NAME = "Churn_DT_Model"

COLUMNS_TO_DROP = [
    'RowNumber', 
    'CustomerId'
]
BINARY_FEATURES = [
    'Gender'
]
ONE_HOT_ENCODE_COLUMNS = [
    'Geography'
]
MODEL_PARAMS = {
    'max_depth': [5],
    'min_samples_split': [10],
    'min_samples_leaf': [4],
    'criterion': ['gini']
}
