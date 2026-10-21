"""Data contract for forecasts with historical and known-future covariates.

This module prepares point-in-time inputs. Model adapters consume ``ForecastWindow``
in a later stage; the existing univariate agent remains backward compatible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


KEY_COLUMNS = ("unique_id", "ds")


@dataclass(frozen=True)
class ForecastWindow:
    """Inputs available at one forecast origin, plus optional held-out targets."""

    origin: pd.Timestamp
    history: pd.DataFrame
    future_covariates: pd.DataFrame
    actuals: pd.DataFrame

    def model_input(self) -> ForecastWindow:
        """Return the model-visible portion, without held-out targets."""
        return ForecastWindow(
            origin=self.origin,
            history=self.history.copy(),
            future_covariates=self.future_covariates.copy(),
            actuals=pd.DataFrame(columns=[*KEY_COLUMNS, "y"]),
        )


@dataclass
class ForecastData:
    """Time series data with explicit roles for each covariate.

    ``history`` contains ``unique_id``, ``ds``, ``y`` and historical covariates.
    ``future_covariates`` contains keys and values assumed known at prediction
    time. No publication-time information is inferred or validated here.
    """

    history: pd.DataFrame
    freq: str
    historical_covariate_cols: tuple[str, ...] = ()
    future_covariate_cols: tuple[str, ...] = ()
    future_covariates: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.historical_covariate_cols = tuple(self.historical_covariate_cols)
        self.future_covariate_cols = tuple(self.future_covariate_cols)
        historical = set(self.historical_covariate_cols)
        future = set(self.future_covariate_cols)
        # The same variable may have observed/archived values before the origin
        # and known forecast values after it (for example, an NWP wind speed).
        # The two roles live in separate frames and may share a column name.
        if (historical | future) & {*KEY_COLUMNS, "y"}:
            raise ValueError("Covariate columns cannot be unique_id, ds, or y")
        if len(historical) != len(self.historical_covariate_cols) or len(future) != len(
            self.future_covariate_cols
        ):
            raise ValueError("Covariate column names must be unique")
        try:
            pd.tseries.frequencies.to_offset(self.freq)
        except ValueError as exc:
            raise ValueError(f"Invalid frequency: {self.freq}") from exc

        self.history = self._prepare(
            self.history, (*KEY_COLUMNS, "y", *self.historical_covariate_cols), "history"
        )
        if self.history.empty:
            raise ValueError("history must contain at least one observation")
        if self.history["y"].isna().any():
            raise ValueError("history.y must not contain missing values")
        try:
            finite_targets = np.isfinite(self.history["y"].to_numpy(dtype=float)).all()
        except (TypeError, ValueError) as exc:
            raise ValueError("history.y must contain finite numeric values") from exc
        if not finite_targets:
            raise ValueError("history.y must contain finite numeric values")

        if self.future_covariate_cols:
            if self.future_covariates is None:
                raise ValueError("future_covariates is required for future covariate columns")
            self.future_covariates = self._prepare(
                self.future_covariates,
                (*KEY_COLUMNS, *self.future_covariate_cols),
                "future_covariates",
            )
            unknown = set(self.future_covariates.unique_id) - set(self.history.unique_id)
            if unknown:
                raise ValueError(f"Unknown series in future_covariates: {sorted(unknown)}")
        elif self.future_covariates is not None:
            raise ValueError("Specify future_covariate_cols when passing future_covariates")

    @staticmethod
    def _prepare(df: pd.DataFrame, required: tuple[str, ...], name: str) -> pd.DataFrame:
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        result = df.loc[:, list(required)].copy()
        result["ds"] = pd.to_datetime(result["ds"], errors="raise")
        if result[list(KEY_COLUMNS)].isna().any().any():
            raise ValueError(f"{name} has missing series IDs or timestamps")
        if result.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(f"{name} has duplicate (unique_id, ds) keys")
        return result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)

    def window(
        self, origin: str | pd.Timestamp, h: int, context_length: int | None = None
    ) -> ForecastWindow:
        """Build a leakage-free origin window for every series.

        Actual targets in the forecast horizon are returned separately for
        scoring; model adapters must only receive ``history`` and
        ``future_covariates``. ``context_length`` keeps at most that many
        observations per series, including the origin. If omitted, all
        available history is returned.
        """
        if h < 1:
            raise ValueError("h must be positive")
        if context_length is not None and context_length < 1:
            raise ValueError("context_length must be positive")
        origin = pd.Timestamp(origin)
        history = self.history.loc[self.history.ds <= origin].copy()
        ids = self.history.unique_id.unique()
        expected_dates = pd.date_range(start=origin, periods=h + 1, freq=self.freq)[1:]
        if not all(
            (history.loc[history.unique_id == uid, "ds"] == origin).any() for uid in ids
        ):
            raise ValueError("Every series must have an observation at the origin")
        if context_length is not None:
            history = history.groupby("unique_id", sort=False).tail(context_length).copy()

        keys = pd.MultiIndex.from_product(
            [ids, expected_dates], names=list(KEY_COLUMNS)
        ).to_frame(index=False)
        if self.future_covariate_cols:
            future = keys.merge(
                self.future_covariates,
                on=list(KEY_COLUMNS),
                how="left",
                validate="one_to_one",
                indicator=True,
            )
            if (future["_merge"] != "both").any() or future[
                list(self.future_covariate_cols)
            ].isna().any().any():
                raise ValueError("Future covariates must cover every series and horizon timestamp")
            future = future.drop(columns="_merge")
        else:
            future = keys

        actuals = keys.merge(
            self.history.loc[:, [*KEY_COLUMNS, "y"]],
            on=list(KEY_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        return ForecastWindow(origin, history, future, actuals)
