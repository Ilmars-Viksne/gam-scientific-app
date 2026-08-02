from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    SplineTransformer,
)
from sklearn.utils.validation import check_is_fitted

FloatArray = NDArray[np.float64]


class GAMFeatureTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """Transform configured predictors into a GAM design matrix.

    The transformer supports:

    - spline bases for smooth predictors;
    - unchanged numeric values for linear predictors;
    - one-hot encoding for categorical predictors;
    - tensor-product interactions between configured smooth predictors.

    Categorical levels are supplied explicitly so that every inner and
    outer cross-validation fold uses the same categorical vocabulary.
    """

    def __init__(
        self,
        *,
        smooth_features: tuple[str, ...] = (),
        linear_features: tuple[str, ...] = (),
        categorical_features: tuple[str, ...] = (),
        categorical_levels: tuple[
            tuple[str, ...],
            ...,
        ] = (),
        interaction_pairs: tuple[
            tuple[str, str],
            ...,
        ] = (),
        missing_policies: tuple[
            tuple[str, str],
            ...,
        ] = (),
        n_knots: int = 3,
        degree: int = 2,
        interaction_scale: float = 1.0,
    ) -> None:
        """Initialize the feature transformer.

        Parameters are assigned without modification because scikit-learn
        must be able to clone the estimator during cross-validation.
        """

        self.smooth_features = smooth_features
        self.linear_features = linear_features
        self.categorical_features = categorical_features
        self.categorical_levels = categorical_levels
        self.interaction_pairs = interaction_pairs
        self.missing_policies = missing_policies
        self.n_knots = n_knots
        self.degree = degree
        self.interaction_scale = interaction_scale

    def _validate(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate the input frame and return columns in fitted order."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError("GAMFeatureTransformer requires a pandas DataFrame.")

        if X.columns.has_duplicates:
            duplicate_columns = X.columns[X.columns.duplicated()].astype(str).tolist()

            raise ValueError(
                f"Input data contains duplicate column names: {duplicate_columns}."
            )

        required_columns = list(
            dict.fromkeys(
                (
                    *self.smooth_features,
                    *self.linear_features,
                    *self.categorical_features,
                )
            )
        )

        missing_columns = [
            feature_name
            for feature_name in required_columns
            if feature_name not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Input data is missing required features: {missing_columns}."
            )

        return X.loc[:, required_columns].copy()

    def _validate_configuration(
        self,
        frame: pd.DataFrame,
    ) -> None:
        all_feature_groups = (
            tuple(self.smooth_features)
            + tuple(self.linear_features)
            + tuple(self.categorical_features)
        )

        if len(all_feature_groups) != len(set(all_feature_groups)):
            raise ValueError("A predictor cannot have more than one feature role.")

        required_features = set(all_feature_groups)

        for left_name, right_name in self.interaction_pairs:
            required_features.add(left_name)
            required_features.add(right_name)

            if left_name == right_name:
                raise ValueError(
                    "An interaction cannot contain the same feature "
                    f"twice: {left_name!r}."
                )

            if left_name not in self.smooth_features:
                raise ValueError(
                    f"Interaction feature {left_name!r} is not configured as smooth."
                )

            if right_name not in self.smooth_features:
                raise ValueError(
                    f"Interaction feature {right_name!r} is not configured as smooth."
                )

        missing_columns = sorted(required_features - set(frame.columns))

        if missing_columns:
            raise ValueError(
                f"Input data is missing configured predictors: {missing_columns}"
            )

        if len(self.categorical_levels) != len(self.categorical_features):
            raise ValueError(
                "A category-level sequence must be supplied for every "
                "categorical feature. "
                f"Received {len(self.categorical_levels)} category "
                "sequences for "
                f"{len(self.categorical_features)} categorical "
                "features."
            )

        for feature_name, levels in zip(
            self.categorical_features,
            self.categorical_levels,
            strict=True,
        ):
            normalized_levels = [str(value) for value in levels]

            if not normalized_levels:
                raise ValueError(
                    f"Categorical feature {feature_name!r} has no "
                    "configured categories."
                )

            if len(normalized_levels) != len(set(normalized_levels)):
                raise ValueError(
                    f"Categorical feature {feature_name!r} contains "
                    "duplicate configured categories."
                )

        valid_missing_policies = {
            "error",
            "median",
            "most_frequent",
        }

        configured_features = set(all_feature_groups)

        for feature_name, policy in self.missing_policies:
            if feature_name not in configured_features:
                raise ValueError(
                    f"Missing-value policy configured for unknown "
                    f"feature {feature_name!r}."
                )

            if policy not in valid_missing_policies:
                raise ValueError(
                    f"Unsupported missing-value policy {policy!r} "
                    f"for feature {feature_name!r}."
                )

    def _validate_required_columns(
        self,
        frame: pd.DataFrame,
    ) -> None:
        required = set(self.smooth_features)
        required.update(self.linear_features)
        required.update(self.categorical_features)

        for left_name, right_name in self.interaction_pairs:
            required.add(left_name)
            required.add(right_name)

        missing = sorted(required - set(frame.columns))

        if missing:
            raise ValueError(
                "Input data is missing predictors required by the "
                f"fitted transformer: {missing}"
            )

    def _fit_missing_values(
        self,
        frame: pd.DataFrame,
    ) -> None:
        numeric_features = tuple(self.smooth_features) + tuple(self.linear_features)

        for feature_name in numeric_features:
            values = pd.to_numeric(
                frame[feature_name],
                errors="coerce",
            )

            policy = self.missing_policy_map_.get(
                feature_name,
                "error",
            )

            if not values.isna().any():
                continue

            if policy == "error":
                raise ValueError(
                    f"Feature {feature_name!r} contains missing or non-numeric values."
                )

            if policy == "median":
                median = values.median()

                if pd.isna(median):
                    raise ValueError(
                        f"Cannot calculate a median for feature {feature_name!r}."
                    )

                self.fill_values_[feature_name] = float(median)
                continue

            if policy == "most_frequent":
                mode = values.mode(dropna=True)

                if mode.empty:
                    raise ValueError(
                        f"Cannot calculate a most-frequent value for "
                        f"feature {feature_name!r}."
                    )

                self.fill_values_[feature_name] = float(mode.iloc[0])
                continue

            raise ValueError(
                f"Missing-value policy {policy!r} is not valid for "
                f"numeric feature {feature_name!r}."
            )

        for feature_name in self.categorical_features:
            values = frame[feature_name].astype("string")

            policy = self.missing_policy_map_.get(
                feature_name,
                "error",
            )

            if not values.isna().any():
                continue

            if policy == "error":
                raise ValueError(
                    f"Categorical feature {feature_name!r} contains missing values."
                )

            if policy != "most_frequent":
                raise ValueError(
                    f"Categorical feature {feature_name!r} requires "
                    "'error' or 'most_frequent' missing-value "
                    "handling."
                )

            mode = values.mode(dropna=True)

            if mode.empty:
                raise ValueError(
                    f"Cannot calculate a most-frequent value for "
                    f"categorical feature {feature_name!r}."
                )

            fill_value = str(mode.iloc[0])
            configured = {
                str(value) for value in self._levels_for_feature(feature_name)
            }

            if fill_value not in configured:
                raise ValueError(
                    f"Most-frequent value {fill_value!r} for feature "
                    f"{feature_name!r} is not present in its "
                    "configured categories."
                )

            self.fill_values_[feature_name] = fill_value

    def _prepare_frame(
        self,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        prepared = frame.copy()

        numeric_features = tuple(self.smooth_features) + tuple(self.linear_features)

        for feature_name in numeric_features:
            values = pd.to_numeric(
                prepared[feature_name],
                errors="coerce",
            )

            if feature_name in self.fill_values_:
                values = values.fillna(self.fill_values_[feature_name])

            if values.isna().any():
                raise ValueError(
                    f"Feature {feature_name!r} contains missing or "
                    "non-numeric values during transformation."
                )

            prepared[feature_name] = values.astype(np.float64)

        for feature_name in self.categorical_features:
            values = prepared[feature_name].astype("string")

            if feature_name in self.fill_values_:
                values = values.fillna(str(self.fill_values_[feature_name]))

            if values.isna().any():
                raise ValueError(
                    f"Categorical feature {feature_name!r} contains "
                    "missing values during transformation."
                )

            prepared[feature_name] = values

        return prepared

    def _categorical_frame(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        frame = X.loc[
            :,
            list(self.categorical_features),
        ].copy()

        for feature_name in self.categorical_features:
            frame[feature_name] = frame[feature_name].astype("string")

        return frame

    def _configured_category_lists(
        self,
    ) -> list[list[str]]:
        category_lists: list[list[str]] = []

        for feature_name, levels in zip(
            self.categorical_features,
            self.categorical_levels,
            strict=True,
        ):
            normalized = [str(value) for value in levels]

            if not normalized:
                raise ValueError(
                    f"Categorical feature {feature_name!r} has no "
                    "configured categories."
                )

            category_lists.append(normalized)

        return category_lists

    def _levels_for_feature(
        self,
        feature_name: str,
    ) -> tuple[str, ...]:
        try:
            index = self.categorical_features.index(feature_name)
        except ValueError as error:
            raise ValueError(f"Feature {feature_name!r} is not categorical.") from error

        return self.categorical_levels[index]

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> GAMFeatureTransformer:
        """Fit all preprocessing components used by the GAM transformer."""

        X = self._validate(X)
        policies = dict(self.missing_policies)

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )
        self.n_features_in_ = X.shape[1]

        self.smooth_imputers_: dict[
            str,
            SimpleImputer | None,
        ] = {}

        self.spline_transformers_: dict[
            str,
            SplineTransformer,
        ] = {}

        self.linear_imputers_: dict[
            str,
            SimpleImputer | None,
        ] = {}

        self.categorical_imputers_: dict[
            str,
            SimpleImputer | None,
        ] = {}

        self.categorical_encoder_: OneHotEncoder | None = None

        #
        # Smooth features
        #

        for feature_name in self.smooth_features:
            values = X.loc[:, [feature_name]]

            missing_policy = policies.get(
                feature_name,
                "error",
            )

            if missing_policy == "error":
                if values[feature_name].isna().any():
                    raise ValueError(
                        f"Smooth feature {feature_name!r} contains missing values."
                    )

                imputer = None
                prepared = values.to_numpy(
                    dtype=np.float64,
                )

            elif missing_policy == "median":
                imputer = SimpleImputer(
                    strategy="median",
                )
                prepared = imputer.fit_transform(values)

            else:
                raise ValueError(
                    f"Unsupported missing-value policy "
                    f"{missing_policy!r} for smooth feature "
                    f"{feature_name!r}."
                )

            spline = SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                include_bias=False,
            )
            spline.fit(prepared)

            self.smooth_imputers_[feature_name] = imputer
            self.spline_transformers_[feature_name] = spline

        #
        # Linear features
        #

        for feature_name in self.linear_features:
            values = X.loc[:, [feature_name]]

            missing_policy = policies.get(
                feature_name,
                "error",
            )

            if missing_policy == "error":
                if values[feature_name].isna().any():
                    raise ValueError(
                        f"Linear feature {feature_name!r} contains missing values."
                    )

                imputer = None

            elif missing_policy == "median":
                imputer = SimpleImputer(
                    strategy="median",
                )
                imputer.fit(values)

            else:
                raise ValueError(
                    f"Unsupported missing-value policy "
                    f"{missing_policy!r} for linear feature "
                    f"{feature_name!r}."
                )

            self.linear_imputers_[feature_name] = imputer

        #
        # Categorical features
        #

        if self.categorical_features:
            if len(self.categorical_levels) != len(self.categorical_features):
                raise ValueError(
                    "A configured category sequence is required for "
                    "every categorical feature. "
                    f"Received {len(self.categorical_levels)} category "
                    f"sequences for {len(self.categorical_features)} "
                    "categorical features."
                )

            configured_categories: list[list[str]] = []

            for feature_name, configured_levels in zip(
                self.categorical_features,
                self.categorical_levels,
                strict=True,
            ):
                categories = [str(value) for value in configured_levels]

                if not categories:
                    raise ValueError(
                        f"Categorical feature {feature_name!r} has no "
                        "configured categories."
                    )

                if len(categories) != len(set(categories)):
                    raise ValueError(
                        f"Categorical feature {feature_name!r} contains "
                        "duplicate configured categories."
                    )

                configured_categories.append(categories)

            categorical_data = X.loc[
                :,
                list(self.categorical_features),
            ].copy()

            for feature_name, allowed_categories in zip(
                self.categorical_features,
                configured_categories,
                strict=True,
            ):
                missing_policy = policies.get(
                    feature_name,
                    "error",
                )

                column = categorical_data.loc[
                    :,
                    [feature_name],
                ]

                if missing_policy == "error":
                    if column[feature_name].isna().any():
                        raise ValueError(
                            f"Categorical feature {feature_name!r} "
                            "contains missing values."
                        )

                    imputer = None

                elif missing_policy == "most_frequent":
                    imputer = SimpleImputer(
                        strategy="most_frequent",
                    )

                    imputed_values = imputer.fit_transform(column).ravel()

                    categorical_data.loc[
                        :,
                        feature_name,
                    ] = imputed_values

                else:
                    raise ValueError(
                        f"Unsupported missing-value policy "
                        f"{missing_policy!r} for categorical feature "
                        f"{feature_name!r}."
                    )

                self.categorical_imputers_[feature_name] = imputer

                categorical_data[feature_name] = categorical_data[feature_name].astype(
                    "string"
                )

                observed_categories = set(
                    categorical_data[feature_name].dropna().astype(str).unique()
                )

                allowed_category_set = set(allowed_categories)

                unknown_categories = sorted(observed_categories - allowed_category_set)

                if unknown_categories:
                    raise ValueError(
                        f"Categorical feature {feature_name!r} contains "
                        "values that are absent from the configured "
                        f"category vocabulary: {unknown_categories}."
                    )

            self.categorical_encoder_ = OneHotEncoder(
                categories=configured_categories,
                drop="first",
                handle_unknown="error",
                sparse_output=False,
                dtype=np.float64,
            )

            self.categorical_encoder_.fit(
                categorical_data.loc[
                    :,
                    list(self.categorical_features),
                ]
            )

        #
        # Interaction validation
        #

        eligible_interaction_features = set(self.smooth_features) | set(
            self.linear_features
        )

        for left_name, right_name in self.interaction_pairs:
            if left_name == right_name:
                raise ValueError(
                    "An interaction pair cannot contain the same "
                    f"feature twice: {left_name!r}."
                )

            unavailable_features = {
                left_name,
                right_name,
            } - eligible_interaction_features

            if unavailable_features:
                raise ValueError(
                    "Interaction pairs may contain only configured "
                    "smooth or linear features. Invalid features: "
                    f"{sorted(unavailable_features)}."
                )

        self.is_fitted_ = True

        #
        # Interaction validation
        #

        for left_name, right_name in self.interaction_pairs:
            if left_name == right_name:
                raise ValueError(
                    "An interaction pair cannot contain the same "
                    f"feature twice: {left_name!r}."
                )

            if left_name not in self.smooth_features:
                raise ValueError(
                    f"Interaction feature {left_name!r} is not configured as smooth."
                )

            if right_name not in self.smooth_features:
                raise ValueError(
                    f"Interaction feature {right_name!r} is not configured as smooth."
                )

        self.output_feature_names_ = np.asarray(
            self._build_feature_names(),
            dtype=object,
        )

        self.is_fitted_ = True

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Transform input data into the fitted GAM design matrix."""

        check_is_fitted(
            self,
            attributes=[
                "feature_names_in_",
                "n_features_in_",
                "smooth_imputers_",
                "spline_transformers_",
                "linear_imputers_",
                "categorical_imputers_",
                "categorical_encoder_",
                "is_fitted_",
            ],
        )

        X = self._validate(X)

        expected_columns = list(self.feature_names_in_)
        missing_columns = [
            str(column) for column in expected_columns if column not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                "Input data is missing columns required by the fitted "
                f"transformer: {missing_columns}."
            )

        transformed_parts: list[np.ndarray] = []

        # Store the transformed representation of every numerical feature.
        # These representations are reused when constructing interactions.
        numeric_feature_matrices: dict[str, np.ndarray] = {}

        #
        # Smooth features
        #

        for feature_name in self.smooth_features:
            values = X.loc[:, [feature_name]]

            imputer = self.smooth_imputers_[feature_name]

            if imputer is None:
                if values[feature_name].isna().any():
                    raise ValueError(
                        f"Smooth feature {feature_name!r} contains missing values."
                    )

                prepared = values.to_numpy(
                    dtype=np.float64,
                )
            else:
                prepared = imputer.transform(values)

            spline = self.spline_transformers_[feature_name]

            transformed = np.asarray(
                spline.transform(prepared),
                dtype=np.float64,
            )

            if transformed.ndim != 2:
                raise ValueError(
                    f"Smooth feature {feature_name!r} produced an "
                    "invalid transformed array."
                )

            if transformed.shape[0] != len(X):
                raise ValueError(
                    f"Smooth feature {feature_name!r} produced an "
                    "unexpected number of rows."
                )

            numeric_feature_matrices[feature_name] = transformed
            transformed_parts.append(transformed)

        #
        # Linear features
        #

        for feature_name in self.linear_features:
            values = X.loc[:, [feature_name]]

            imputer = self.linear_imputers_[feature_name]

            if imputer is None:
                if values[feature_name].isna().any():
                    raise ValueError(
                        f"Linear feature {feature_name!r} contains missing values."
                    )

                transformed = values.to_numpy(
                    dtype=np.float64,
                )
            else:
                transformed = np.asarray(
                    imputer.transform(values),
                    dtype=np.float64,
                )

            if transformed.ndim == 1:
                transformed = transformed.reshape(-1, 1)

            if transformed.ndim != 2:
                raise ValueError(
                    f"Linear feature {feature_name!r} produced an "
                    "invalid transformed array."
                )

            if transformed.shape[0] != len(X):
                raise ValueError(
                    f"Linear feature {feature_name!r} produced an "
                    "unexpected number of rows."
                )

            numeric_feature_matrices[feature_name] = transformed
            transformed_parts.append(transformed)

        #
        # Categorical features
        #

        if self.categorical_features:
            if self.categorical_encoder_ is None:
                raise RuntimeError("The categorical encoder has not been fitted.")

            categorical_data = X.loc[
                :,
                list(self.categorical_features),
            ].copy()

            for feature_name in self.categorical_features:
                imputer = self.categorical_imputers_[feature_name]

                column = categorical_data.loc[
                    :,
                    [feature_name],
                ]

                if imputer is None:
                    if column[feature_name].isna().any():
                        raise ValueError(
                            f"Categorical feature {feature_name!r} "
                            "contains missing values."
                        )
                else:
                    imputed_values = imputer.transform(column).ravel()

                    categorical_data.loc[
                        :,
                        feature_name,
                    ] = imputed_values

                # The categories supplied during fit are strings.
                # Transform-time values must use the same representation.
                categorical_data[feature_name] = categorical_data[feature_name].astype(
                    "string"
                )

            try:
                categorical_matrix = np.asarray(
                    self.categorical_encoder_.transform(
                        categorical_data.loc[
                            :,
                            list(self.categorical_features),
                        ]
                    ),
                    dtype=np.float64,
                )
            except ValueError as error:
                raise ValueError(
                    "Categorical transformation failed. The input may "
                    "contain category values that are absent from the "
                    "configured category vocabulary."
                ) from error

            if categorical_matrix.ndim != 2:
                raise ValueError(
                    "Categorical encoding produced an invalid transformed array."
                )

            if categorical_matrix.shape[0] != len(X):
                raise ValueError(
                    "Categorical encoding produced an unexpected number of rows."
                )

            transformed_parts.append(categorical_matrix)

        #
        # Pairwise numerical interactions
        #

        for left_name, right_name in self.interaction_pairs:
            if left_name not in numeric_feature_matrices:
                raise ValueError(
                    f"Interaction feature {left_name!r} has no fitted "
                    "numerical representation."
                )

            if right_name not in numeric_feature_matrices:
                raise ValueError(
                    f"Interaction feature {right_name!r} has no fitted "
                    "numerical representation."
                )

            left_matrix = numeric_feature_matrices[left_name]
            right_matrix = numeric_feature_matrices[right_name]

            # For each observation, calculate the tensor product of the
            # transformed left and right feature representations.
            #
            # Shapes:
            #     left_matrix:       (rows, left_basis_count)
            #     right_matrix:      (rows, right_basis_count)
            #     interaction_cube:  (rows, left_basis_count,
            #                         right_basis_count)
            interaction_cube = np.einsum(
                "ij,ik->ijk",
                left_matrix,
                right_matrix,
            )

            interaction_matrix = interaction_cube.reshape(
                len(X),
                -1,
            )

            interaction_matrix = interaction_matrix * float(self.interaction_scale)

            transformed_parts.append(interaction_matrix)

        #
        # Assemble the complete design matrix
        #

        if not transformed_parts:
            return np.empty(
                shape=(len(X), 0),
                dtype=np.float64,
            )

        design_matrix = np.column_stack(transformed_parts).astype(
            np.float64,
            copy=False,
        )

        if design_matrix.shape[0] != len(X):
            raise ValueError(
                "The transformed design matrix has an unexpected number of rows."
            )

        if not np.isfinite(design_matrix).all():
            raise ValueError("The transformed design matrix contains nonfinite values.")

        return design_matrix

    def _build_feature_names(
        self,
    ) -> list[str]:
        names: list[str] = []

        for feature_name in self.smooth_features:
            transformer = self.spline_transformers_[feature_name]
            basis_count = transformer.n_features_out_

            names.extend(
                f"main_spline__{feature_name}__basis_{index}"
                for index in range(basis_count)
            )

        names.extend(
            f"main_linear__{feature_name}" for feature_name in self.linear_features
        )

        if self.categorical_features:
            if self.categorical_encoder_ is None:
                raise RuntimeError("The categorical encoder was not fitted.")

            encoded_names = self.categorical_encoder_.get_feature_names_out(
                self.categorical_features
            )

            names.extend(
                f"main_categorical__{encoded_name}" for encoded_name in encoded_names
            )

        for left_name, right_name in self.interaction_pairs:
            left_count = self.spline_transformers_[left_name].n_features_out_

            right_count = self.spline_transformers_[right_name].n_features_out_

            names.extend(
                (
                    f"interaction__{left_name}:{right_name}__basis_{left_index}:{right_index}"
                )
                for left_index in range(left_count)
                for right_index in range(right_count)
            )

        return names

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> NDArray[np.object_]:
        """Return transformed feature names."""

        del input_features

        check_is_fitted(
            self,
            attributes=["output_feature_names_"],
        )

        return self.output_feature_names_.copy()

    @staticmethod
    def _tensor_product(
        left: FloatArray,
        right: FloatArray,
    ) -> FloatArray:
        if left.shape[0] != right.shape[0]:
            raise ValueError(
                "Interaction matrices must contain the same number of rows."
            )

        product = np.einsum(
            "ij,ik->ijk",
            left,
            right,
        )

        return np.asarray(
            product.reshape(
                left.shape[0],
                left.shape[1] * right.shape[1],
            ),
            dtype=np.float64,
        )
