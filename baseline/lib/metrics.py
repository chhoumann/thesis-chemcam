import numpy as np
from sklearn.metrics import mean_squared_error


def rmse_metric(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def std_dev_metric(y_true, y_pred):
    return float(np.std(y_true - y_pred, ddof=1))

def error_consistency(y_true, y_pred):
    abs_errors = np.abs(y_true - y_pred)
    mean_abs_error = np.mean(abs_errors)
    squared_diffs = (abs_errors - mean_abs_error) ** 2
    error_consistency = np.sqrt(np.sum(squared_diffs) / (len(y_true) - 1))
    
    return error_consistency
