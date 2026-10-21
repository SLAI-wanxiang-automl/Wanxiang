"""Split boundaries, tuning isolation, and frozen fitting contracts."""
import numpy as np
import pandas as pd
import pytest

from eneygyexpert.backtesting import SeasonalNaiveWindowModel
from eneygyexpert.data import ForecastData
from eneygyexpert.pipeline import automatic_window_count, train_validation_test
from eneygyexpert.tuning import GridTunedWindowModel


class AuditedSeasonal(SeasonalNaiveWindowModel):
    def __init__(self, seasonality, alias, fits):
        super().__init__(seasonality, alias)
        self.fits = fits

    def fit_frozen(self, window, freq):
        self.fits.append((self.alias, len(window.history), window.origin))
        assert window.actuals.empty
        return self


def test_split_tune_and_fit_once():
    dates = pd.date_range('2020', periods=100, freq='h')
    data = ForecastData(pd.DataFrame(dict(unique_id=['a'] * 100, ds=dates,
                                         y=np.tile([0., 1., 2., 3.], 25))), freq='h')
    fits = []
    grid = GridTunedWindowModel([
        AuditedSeasonal(4, 'period4', fits),
        AuditedSeasonal(3, 'period3', fits),
    ], alias='Tuned', inner_h=4)
    result = train_validation_test(data, [grid], h=4, min_train_size=8)
    assert result.inner_n_windows == 13
    assert result.boundaries['train_end'] == dates[59]
    assert result.boundaries['test_start'] == dates[80]
    assert result.fixed_models['Tuned'].selected.alias == 'period4'
    assert fits == [('period4', 60, dates[59]), ('period4', 80, dates[79])]
    assert result.validation['Tuned'].forecasts.origin.nunique() == 5
    assert result.test.forecasts.origin.nunique() == 5
    assert result.test.forecasts.y_pred.equals(result.test.forecasts.y)
    changed = data.history.copy()
    changed.loc[changed.ds >= dates[60], 'y'] += 1000
    altered = train_validation_test(ForecastData(changed, freq='h'), [grid], h=4,
                                    min_train_size=8)
    assert altered.fixed_models['Tuned'].selected.alias == 'period4'


def test_window_count():
    assert automatic_window_count(60, 4, 4, 8) == 13
    assert automatic_window_count(21, 4, 4) == 5


def test_default_median_adapter_keeps_candidate_and_prediction_ensemble_results():
    from eneygyexpert.rolling_models.ensemble import MedianEnsembleAdapter

    dates = pd.date_range('2020', periods=100, freq='h')
    data = ForecastData(pd.DataFrame(dict(unique_id=['a'] * 100, ds=dates,
                                         y=np.tile([0., 1., 2., 3.], 25))), freq='h')
    ensemble = MedianEnsembleAdapter([
        SeasonalNaiveWindowModel(4, 'period4'),
        SeasonalNaiveWindowModel(3, 'period3'),
    ])
    result = train_validation_test(data, [ensemble], h=4, min_train_size=8)
    assert result.selected_model == ensemble.alias == 'MedianModelEnsemble'
    assert set(result.test_results) == {
        'MedianModelEnsemble', 'MedianEnsemble', 'WeightedEnsemble'
    }
    assert result.test is result.test_results[ensemble.alias]
    assert result.test.summary.iloc[0].weight_policy == 'frozen'
    assert result.test_results['MedianEnsemble'].summary.iloc[0].weight_policy == (
        'prediction_ensemble'
    )


@pytest.mark.parametrize('alias', ['MedianEnsemble', 'WeightedEnsemble'])
def test_prediction_ensemble_aliases_remain_reserved(alias):
    dates = pd.date_range('2020', periods=100, freq='h')
    data = ForecastData(pd.DataFrame(dict(unique_id=['a'] * 100, ds=dates,
                                         y=np.zeros(100))), freq='h')
    with pytest.raises(ValueError, match='reserved aliases'):
        train_validation_test(data, [SeasonalNaiveWindowModel(4, alias)], h=4)


def test_compare_test_models_and_plot(tmp_path):
    from eneygyexpert.rolling_models.plotting import plot_test_windows
    from eneygyexpert.rolling_models.report import save_pipeline_result

    dates = pd.date_range('2020', periods=100, freq='h')
    data = ForecastData(pd.DataFrame(dict(unique_id=['a'] * 100, ds=dates,
                                         y=np.tile([0., 1., 2., 3.], 25))), freq='h')
    fits = []
    result = train_validation_test(
        data, [AuditedSeasonal(4, 'period4', fits), AuditedSeasonal(3, 'period3', fits)],
        h=4, min_train_size=8, top_k=2,
    )
    assert result.selected_model == 'period4'
    assert result.test is result.test_results['period4']
    assert [(alias, size) for alias, size, _ in fits] == [
        ('period4', 60), ('period3', 60), ('period4', 80), ('period3', 80)]
    save_pipeline_result(result, tmp_path)
    assert len(pd.read_csv(tmp_path / 'test_comparison_scores.csv')) == 4
    plot_test_windows(data, result.test_results, tmp_path, n_windows=3, seed=42,
                      context_length=8)
    selected = pd.read_csv(tmp_path / 'plotted_test_windows.csv')
    assert len(selected) == 3
    assert (pd.to_datetime(selected.origin) >= dates[79]).all()
    assert (tmp_path / 'test_windows.png').stat().st_size > 1000
    plot_test_windows(data, result.test_results, tmp_path, n_windows=3, seed=42,
                      context_length=8)
    pd.testing.assert_frame_equal(selected, pd.read_csv(tmp_path / 'plotted_test_windows.csv'))
    assert len(fits) == 4
    plot_test_windows(data, result.test_results, tmp_path, n_windows=20)
    assert len(pd.read_csv(tmp_path / 'plotted_test_windows.csv')) == 5


def test_top_k_excludes_losers_and_validation_weights():
    from dataclasses import replace
    from eneygyexpert.rolling_models.prediction_ensemble import ensemble_test_results

    dates = pd.date_range('2020', periods=100, freq='h')
    data = ForecastData(pd.DataFrame(dict(unique_id=['a'] * 100, ds=dates,
                                         y=np.tile([0., 1., 2., 3.], 25))), freq='h')
    fits = []
    models = [AuditedSeasonal(4, 'best', fits), AuditedSeasonal(3, 'other', fits),
              AuditedSeasonal(2, 'third', fits)]
    result = train_validation_test(data, models, h=4, min_train_size=8, top_k=2)
    assert len(result.selected_models) == 2
    assert len([f for f in fits if f[1] == 80]) == 2
    loser = (set(result.fixed_models) - set(result.selected_models)).pop()
    assert loser not in result.test_results
    assert result.ensemble_weights['best'] == 1
    assert result.test_results['WeightedEnsemble'].forecasts.y_pred.equals(
        result.test.forecasts.sort_values(['unique_id', 'origin', 'ds', 'lead'])
        .reset_index(drop=True).y_pred)
    members = {a: result.test_results[a] for a in result.selected_models}
    validation = {a: replace(result.validation[a], summary=pd.DataFrame({'mae': [e]}))
                  for a, e in zip(result.selected_models, [1., 3.])}
    ensembles, weights = ensemble_test_results(data, members, validation)
    np.testing.assert_allclose(list(weights.values()), [.75, .25])
    matrix = np.stack([m.forecasts.y_pred for m in members.values()])
    np.testing.assert_allclose(ensembles['WeightedEnsemble'].forecasts.y_pred,
                               [.75, .25] @ matrix)
    np.testing.assert_allclose(ensembles['MedianEnsemble'].forecasts.y_pred,
                               np.median(matrix, axis=0))
    validation = {a: replace(v, summary=pd.DataFrame({'mae': [0.]}))
                  for a, v in validation.items()}
    _, weights = ensemble_test_results(data, members, validation)
    np.testing.assert_allclose(list(weights.values()), [.5, .5])
