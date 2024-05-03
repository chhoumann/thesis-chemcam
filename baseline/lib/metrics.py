import numpy as np
from sklearn.metrics import mean_squared_error


def rmse_metric(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def std_dev_metric(y_true, y_pred):
    return float(np.std(y_true - y_pred))
