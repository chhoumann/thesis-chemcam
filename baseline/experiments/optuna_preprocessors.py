from optuna import Trial
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer


def instantiate_robust_scaler(trial: Trial) -> RobustScaler:
    params = { 
        'quantile_range': trial.suggest_categorical('quantile_range', [
                (25.0, 75.0), 
                (10.0, 90.0), 
                (5.0, 95.0), 
                (35, 65), 
                (30, 70), 
                (40, 60)
            ]
        ),
        'with_centering': trial.suggest_categorical('with_centering', [True, False]),
    }

    return RobustScaler(**params)

def instantiate_standard_scaler(trial: Trial) -> StandardScaler:
    params = { 
        'with_mean': trial.suggest_categorical('with_mean', [True, False]),
        'with_std': trial.suggest_categorical('with_std', [True, False]),
    }

    return StandardScaler(**params)

def instantiate_min_max_scaler(trial: Trial) -> MinMaxScaler:
    params = { 
        'feature_range': trial.suggest_categorical('feature_range', [(0, 1), (-1, 1)]),
    }

    return MinMaxScaler(**params)

def instantiate_power_transformer(trial: Trial) -> PowerTransformer:
    params = { 
        'method': trial.suggest_categorical('method', ['yeo-johnson', 'box-cox']),
        'standardize': trial.suggest_categorical('standardize', [True, False]),
    }

    return PowerTransformer(**params)