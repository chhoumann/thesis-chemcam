import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class Stacker(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimators, final_estimator):
        self.base_estimators = base_estimators
        self.final_estimator = final_estimator
        self.meta_features_ = None

    def fit_cv(self, X, y, fold_indices):
        meta_features = np.zeros((X.shape[0], len(self.base_estimators)))

        for i, (train_idx, test_idx) in enumerate(fold_indices):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for j, (name, estimator) in enumerate(self.base_estimators):
                estimator.fit(X_train, y_train)
                meta_features[test_idx, j] = estimator.predict(X_test)

        self.meta_features_ = meta_features
        self.final_estimator.fit(meta_features, y)
        return self

    def fit(self, X, y):
        for _, estimator in self.base_estimators:
            estimator.fit(X, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack(
            [estimator.predict(X) for name, estimator in self.base_estimators]
        )
        return self.final_estimator.predict(meta_features)

    def cv(self, fold_indices, meta_features, y_full, final_estimator, metric_fns):
        cv_metrics = []

        for train_idx, test_idx in fold_indices:
            X_train, X_test = meta_features[train_idx], meta_features[test_idx]
            y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

            final_estimator.fit(X_train, y_train)
            y_pred = final_estimator.predict(X_test)

            fold_metrics = [metric_fn(y_test, y_pred) for metric_fn in metric_fns]
            cv_metrics.append(fold_metrics)

        return cv_metrics

