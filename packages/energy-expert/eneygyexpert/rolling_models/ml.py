"""The eight original MLForecast Auto models, sharing frozen prediction."""

from ..native_automl import AutoMLForecastWindowAdapter

ML_MODELS = {
    "AutoLinearRegression": "AutoLinearRegression",
    "AutoRidge": "AutoRidge",
    "AutoLasso": "AutoLasso",
    "AutoElasticNet": "AutoElasticNet",
    "AutoRandomForest": "AutoRandomForest",
    "AutoLGBM": "AutoLightGBM",
    "AutoXGBoost": "AutoXGBoost",
    "AutoCatboost": "AutoCatboost",
}


def create_ml_model(name, *, season_length, alias=None, **kwargs):
    """Keep native AutoML search spaces; first version takes only target y."""
    import mlforecast.auto as auto

    if name not in ML_MODELS:
        raise ValueError(f"Unknown ML model {name}")
    return AutoMLForecastWindowAdapter(
        getattr(auto, ML_MODELS[name])(),
        alias=alias or name,
        season_length=season_length,
        **kwargs,
    )
