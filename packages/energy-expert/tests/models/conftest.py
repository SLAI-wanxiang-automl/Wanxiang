import pytest

from eneygyexpert.models.prophet import Prophet
from eneygyexpert.models.stats import (
    ADIDA,
    AutoARIMA,
    SeasonalNaive,
    ZeroModel,
)

models = [
    AutoARIMA(),
    SeasonalNaive(),
    ZeroModel(),
    ADIDA(),
    Prophet(),
]
