from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted


class GAMFeatureTransformer(BaseEstimator, TransformerMixin):
    """Universal GAM design transformer with optional tensor-product interactions."""

    def __init__(
        self,
        smooth_features: tuple[str, ...],
        linear_features: tuple[str, ...],
        categorical_features: tuple[str, ...],
        interaction_pairs: tuple[tuple[str, str], ...] = (),
        missing_policies: tuple[tuple[str, str], ...] = (),
        n_knots: int = 4,
        degree: int = 2,
        interaction_scale: float = 1.0,
    ) -> None:
        self.smooth_features = smooth_features
        self.linear_features = linear_features
        self.categorical_features = categorical_features
        self.interaction_pairs = interaction_pairs
        self.missing_policies = missing_policies
        self.n_knots = n_knots
        self.degree = degree
        self.interaction_scale = interaction_scale

    def fit(self, X: pd.DataFrame, y: Any = None) -> GAMFeatureTransformer:
        X = self._validate(X)
        policies = dict(self.missing_policies)
        numeric = (*self.smooth_features, *self.linear_features)
        self.numeric_imputers_ = {}
        X_work = X.copy()
        for name in numeric:
            policy = policies.get(name, "error")
            if policy == "error":
                if X_work[name].isna().any():
                    raise ValueError(f"Missing values in {name!r}.")
                continue
            strategy = "median" if policy == "median" else "most_frequent"
            imputer = SimpleImputer(strategy=strategy)
            X_work[[name]] = imputer.fit_transform(X_work[[name]])
            self.numeric_imputers_[name] = imputer
        self.categorical_imputers_ = {}
        for name in self.categorical_features:
            policy = policies.get(name, "error")
            if policy == "error":
                if X_work[name].isna().any():
                    raise ValueError(f"Missing values in {name!r}.")
            else:
                imputer = SimpleImputer(strategy="most_frequent")
                X_work[[name]] = imputer.fit_transform(X_work[[name]])
                self.categorical_imputers_[name] = imputer
        self.spline_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            knots="quantile",
            extrapolation="constant",
            include_bias=False,
        ).fit(X_work.loc[:, self.smooth_features].astype(float))
        self.scaler_ = StandardScaler()
        if self.linear_features:
            self.scaler_.fit(X_work.loc[:, self.linear_features].astype(float))
        self.encoder_ = OneHotEncoder(
            drop="first", handle_unknown="ignore", sparse_output=False
        )
        if self.categorical_features:
            self.encoder_.fit(X_work.loc[:, self.categorical_features].astype(str))
        total = int(self.spline_.n_features_out_)
        if total % len(self.smooth_features) != 0:
            raise RuntimeError(
                "Spline outputs do not divide evenly by smooth features."
            )
        self.basis_count_ = total // len(self.smooth_features)
        self.smooth_index_ = {name: i for i, name in enumerate(self.smooth_features)}
        self.feature_names_out_ = np.asarray(self._names(), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["spline_", "feature_names_out_"])
        X_work = self._apply_imputation(self._validate(X))
        smooth = self.spline_.transform(
            X_work.loc[:, self.smooth_features].astype(float)
        )
        blocks = [np.asarray(smooth, dtype=float)]
        if self.linear_features:
            blocks.append(
                self.scaler_.transform(
                    X_work.loc[:, self.linear_features].astype(float)
                )
            )
        if self.categorical_features:
            blocks.append(
                self.encoder_.transform(
                    X_work.loc[:, self.categorical_features].astype(str)
                )
            )
        for left, right in self.interaction_pairs:
            left_basis = self._block(smooth, left)
            right_basis = self._block(smooth, right)
            block = np.einsum("ni,nj->nij", left_basis, right_basis).reshape(
                len(X_work), -1
            )
            blocks.append(block * float(self.interaction_scale))
        result = np.hstack(blocks).astype(float, copy=False)
        if result.shape[1] != len(self.feature_names_out_):
            raise RuntimeError("Transformed feature count mismatch.")
        return result

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, ["feature_names_out_"])
        return self.feature_names_out_.copy()

    def _apply_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for name, imputer in self.numeric_imputers_.items():
            X[[name]] = imputer.transform(X[[name]])
        for name, imputer in self.categorical_imputers_.items():
            X[[name]] = imputer.transform(X[[name]])
        return X

    def _block(self, smooth: np.ndarray, name: str) -> np.ndarray:
        index = self.smooth_index_[name]
        start = index * self.basis_count_
        return smooth[:, start : start + self.basis_count_]

    def _names(self) -> list[str]:
        names = [
            f"main_spline__{name}__basis_{basis}"
            for name in self.smooth_features
            for basis in range(self.basis_count_)
        ]
        names.extend(f"main_linear__{name}" for name in self.linear_features)
        if self.categorical_features:
            names.extend(
                f"main_categorical__{name}"
                for name in self.encoder_.get_feature_names_out(
                    self.categorical_features
                )
            )
        names.extend(
            f"interaction__{left}:{right}__basis_{a}:{b}"
            for left, right in self.interaction_pairs
            for a in range(self.basis_count_)
            for b in range(self.basis_count_)
        )
        return names

    def _validate(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("GAMFeatureTransformer requires a pandas DataFrame.")
        expected = [
            *self.smooth_features,
            *self.linear_features,
            *self.categorical_features,
        ]
        missing = sorted(set(expected) - set(X.columns))
        if missing:
            raise ValueError(f"Missing input features: {missing}")
        if len(expected) != len(set(expected)):
            raise ValueError("Feature roles overlap.")
        smooth = set(self.smooth_features)
        for left, right in self.interaction_pairs:
            if left == right or left not in smooth or right not in smooth:
                raise ValueError(
                    f"Interactions require distinct smooth features: {left}:{right}"
                )
        return X.loc[:, expected].copy()
