from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import load_config
from .data import infer_role, load_table, profile_data, save_profile
from .inspection import inspect_model, verify_link
from .io_utils import read_json, write_yaml_atomic
from .logistic import extract_class_score_parameters
from .reporting import create_reports
from .run_store import FileRunStore
from .workflow import create_run, execute_run


def _preset(name: str) -> dict:
    presets = {
        "quick": {
            "outer_splits": 3,
            "outer_repeats": 1,
            "inner_splits": 3,
            "n_knots": [3],
            "degree": [2],
            "C": [0.1, 1.0, 10.0],
        },
        "standard": {
            "outer_splits": 5,
            "outer_repeats": 3,
            "inner_splits": 5,
            "n_knots": [3, 4, 5],
            "degree": [2, 3],
            "C": [0.01, 0.1, 1.0, 10.0],
        },
        "thorough": {
            "outer_splits": 5,
            "outer_repeats": 5,
            "inner_splits": 5,
            "n_knots": [3, 4, 5, 6],
            "degree": [2, 3],
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        },
    }
    return presets[name]


def command_profile(args) -> None:
    profile = profile_data(args.data, args.target)
    save_profile(profile, args.output)
    print(json.dumps(profile, indent=2))


def _ask(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def command_configure(args) -> None:
    frame = load_table(args.data)

    if args.target not in frame.columns:
        raise ValueError(f"Target {args.target!r} is absent.")

    features: dict[str, dict[str, Any]] = {}

    for name in frame.columns:
        if name == args.target:
            continue

        recommended, reason = infer_role(frame[name])
        role = recommended

        if not args.non_interactive:
            print(f"\n{name}: {reason}; recommended role={recommended}")

            role = _ask(
                "Role (smooth/linear/categorical/exclude)",
                recommended,
            )

        if role not in {
            "smooth",
            "linear",
            "categorical",
            "exclude",
        }:
            raise ValueError(f"Invalid role {role!r} selected for feature {name!r}.")

        spec: dict[str, Any] = {
            "role": role,
            "missing": "error",
        }

        if role == "categorical":
            category_values = sorted(
                frame[name].dropna().unique(),
                key=str,
            )

            spec["categories"] = [str(value) for value in category_values]

        features[name] = spec

    preset = args.preset

    if not args.non_interactive:
        preset = _ask(
            "Search preset (quick/standard/thorough)",
            preset,
        )

    if preset not in {
        "quick",
        "standard",
        "thorough",
    }:
        raise ValueError(f"Invalid search preset: {preset!r}.")

    preset_values: dict[str, Any] = _preset(preset)

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment": {
            "name": args.name or args.data.stem,
            "primary_metric": "log_loss",
        },
        "data": {
            "path": str(args.data.resolve()),
            "target": args.target,
            "row_id": args.row_id,
        },
        "features": features,
        "models": [
            {
                "id": "gam_main",
                "interactions": "none",
            },
            {
                "id": "gam_pairwise",
                "interactions": "all_eligible",
            },
        ],
        "validation": {
            "outer_splits": preset_values["outer_splits"],
            "outer_repeats": preset_values["outer_repeats"],
            "inner_splits": preset_values["inner_splits"],
            "random_state": 42,
        },
        "search": {
            "n_knots": preset_values["n_knots"],
            "degree": preset_values["degree"],
            "C": preset_values["C"],
            "interaction_scale": [
                0.5,
                1.0,
            ],
        },
        "execution": {
            "workers": 1,
            "checkpoint_unit": "outer_fold",
            "stop_on_convergence_warning": True,
        },
    }

    write_yaml_atomic(
        args.output,
        payload,
    )

    print(f"Configuration written to {args.output.resolve()}")


def command_plan(args) -> None:
    config = load_config(args.config)
    main = len(config.search.n_knots) * len(config.search.degree) * len(config.search.C)
    pairwise = main * len(config.search.interaction_scale)
    outer = config.validation.outer_splits * config.validation.outer_repeats
    rows = []
    for model in config.models:
        candidates = main if model.interactions == "none" else pairwise
        fits = (
            candidates * config.validation.inner_splits * outer
            + candidates * config.validation.inner_splits
        )
        rows.append(
            {"model": model.id, "candidates": candidates, "estimated_fits": fits}
        )
    print(pd.DataFrame(rows).to_string(index=False))


def command_run(args) -> None:
    run = create_run(args.config, args.workspace)
    print(f"Run directory: {run.resolve()}")
    execute_run(run)


def command_resume(args) -> None:
    for marker in ["PAUSE", "CANCEL"]:
        path = args.run / "control" / marker
        if path.exists():
            path.unlink()
    execute_run(args.run)


def command_status(args) -> None:
    while True:
        print(json.dumps(read_json(args.run / "status.json"), indent=2))
        if not args.follow:
            break
        state = read_json(args.run / "status.json").get("state")
        if state in {"completed", "failed", "cancelled"}:
            break
        time.sleep(2)


def command_control(args, marker: str) -> None:
    path = args.run / "control" / marker
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    print(f"{marker} requested for {args.run}")


def command_inspect(args) -> None:
    model = args.run / "models" / args.model / "model.joblib"
    output = args.run / "results" / args.model / "inspection"
    inspect_model(model, output, args.reference_class)
    config = load_config(args.run / "config.yaml")
    store = FileRunStore(args.run)
    create_reports(config, store)
    print(f"Inspection written to {output.resolve()}")
    print(f"HTML report regenerated at {(store.reports / 'report.html').resolve()}")


def command_verify_link(args) -> None:
    config = load_config(args.run / "config.yaml")
    frame = load_table(config.data_path)
    active = [name for name, spec in config.features.items() if spec.role != "exclude"]
    X = frame[active].copy()
    for name, spec in config.features.items():
        if spec.role == "categorical" and name in X:
            X[name] = X[name].astype("string")
    model = args.run / "models" / args.model / "model.joblib"
    output = args.run / "results" / args.model / "link_verification"
    error = verify_link(model, X, output)
    store = FileRunStore(args.run)
    create_reports(config, store)
    print(f"Maximum softmax reconstruction error: {error:.17g}")
    print(f"HTML report regenerated at {(store.reports / 'report.html').resolve()}")


def command_compare(args) -> None:
    left = pd.read_csv(args.left / "results" / args.left_model / "fold_metrics.csv")
    right = pd.read_csv(args.right / "results" / args.right_model / "fold_metrics.csv")
    merged = left.merge(
        right,
        on=["repeat", "fold"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    metrics = ["log_loss", "accuracy", "balanced_accuracy", "macro_f1"]
    for metric in metrics:
        merged[f"{metric}_difference"] = (
            merged[f"{metric}_right"] - merged[f"{metric}_left"]
        )
    args.output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output / "comparison.csv", index=False)
    merged[[f"{metric}_difference" for metric in metrics]].agg(
        ["mean", "std", "median"]
    ).T.to_csv(args.output / "summary.csv")
    print((args.output / "summary.csv").read_text(encoding="utf-8"))


def _component_type(
    component_name: str,
) -> str:
    """Classify a transformed GAM component."""

    if component_name.startswith("main_spline__"):
        return "smooth"

    if component_name.startswith("main_linear__"):
        return "linear"

    if component_name.startswith("main_categorical__"):
        return "categorical"

    if component_name.startswith("interaction__"):
        return "interaction"

    if component_name == "intercept":
        return "intercept"

    return "other"


def _component_group(
    component_name: str,
) -> str:
    """Return the predictor-level group for a GAM component."""

    if component_name == "intercept":
        return "intercept"

    if component_name.startswith("main_spline__"):
        remainder = component_name.removeprefix("main_spline__")

        return remainder.split(
            "__basis_",
            maxsplit=1,
        )[0]

    if component_name.startswith("main_linear__"):
        return component_name.removeprefix("main_linear__")

    if component_name.startswith("main_categorical__"):
        return component_name.removeprefix("main_categorical__")

    if component_name.startswith("interaction__"):
        remainder = component_name.removeprefix("interaction__")

        return remainder.split(
            "__basis_",
            maxsplit=1,
        )[0]

    return component_name


def command_predict(args) -> None:
    """Generate detailed predictions for predictor scenarios."""

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    if not input_path.is_file():
        raise FileNotFoundError(f"Prediction input file does not exist: {input_path}")

    model = joblib.load(model_path)

    if not hasattr(model, "named_steps"):
        raise ValueError("The loaded model is not a compatible fitted pipeline.")

    if "features" not in model.named_steps:
        raise ValueError("The loaded pipeline does not contain a 'features' step.")

    if "classifier" not in model.named_steps:
        raise ValueError("The loaded pipeline does not contain a 'classifier' step.")

    transformer = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    if not hasattr(transformer, "feature_names_in_"):
        raise ValueError(
            "The fitted feature transformer does not expose feature_names_in_."
        )

    if not hasattr(classifier, "classes_"):
        raise ValueError("The fitted classifier does not expose target classes.")

    scenarios = pd.read_csv(input_path)

    if scenarios.empty:
        raise ValueError("The prediction input file contains no observations.")

    if scenarios.columns.has_duplicates:
        duplicate_columns = (
            scenarios.columns[scenarios.columns.duplicated()].astype(str).tolist()
        )

        raise ValueError(
            f"Prediction data contains duplicate column names: {duplicate_columns}."
        )

    required_columns = [str(value) for value in transformer.feature_names_in_]

    input_columns = {str(value) for value in scenarios.columns}

    required_column_set = set(required_columns)

    missing_columns = sorted(required_column_set - input_columns)

    if missing_columns:
        raise ValueError(
            f"Prediction data is missing required predictors: {missing_columns}."
        )

    extra_columns = sorted(input_columns - required_column_set)

    if extra_columns:
        print(
            "Extra input columns will be preserved in the output "
            "but excluded from model prediction: "
            f"{extra_columns}",
            flush=True,
        )

    original_scenarios = scenarios.copy()

    model_scenarios = scenarios.loc[
        :,
        required_columns,
    ].copy()

    probabilities = np.asarray(
        model.predict_proba(model_scenarios),
        dtype=np.float64,
    )

    predicted_classes = np.asarray(
        model.predict(model_scenarios),
        dtype=object,
    )

    raw_scores = np.asarray(
        model.decision_function(model_scenarios),
        dtype=np.float64,
    )

    classes = [str(value) for value in classifier.classes_]

    if len(classes) < 2:
        raise ValueError("The classifier must contain at least two classes.")

    if raw_scores.ndim == 1:
        if len(classes) != 2:
            raise ValueError(
                "One-dimensional decision scores are valid only "
                "for binary classification."
            )

        score_matrix = np.column_stack(
            [
                np.zeros_like(raw_scores),
                raw_scores,
            ]
        )

    elif raw_scores.ndim == 2:
        score_matrix = raw_scores

    else:
        raise ValueError(
            "The classifier returned decision scores with an "
            f"unsupported shape: {raw_scores.shape}."
        )

    expected_output_shape = (
        len(model_scenarios),
        len(classes),
    )

    if probabilities.shape != expected_output_shape:
        raise ValueError(
            "The probability matrix has an unexpected shape. "
            f"Received {probabilities.shape}; "
            f"expected {expected_output_shape}."
        )

    if score_matrix.shape != expected_output_shape:
        raise ValueError(
            "The decision-score matrix has an unexpected shape. "
            f"Received {score_matrix.shape}; "
            f"expected {expected_output_shape}."
        )

    expected_prediction_shape = (len(model_scenarios),)

    if predicted_classes.shape != expected_prediction_shape:
        raise ValueError(
            "The predicted-class array has an unexpected shape. "
            f"Received {predicted_classes.shape}; "
            f"expected {expected_prediction_shape}."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError("The predicted probabilities contain nonfinite values.")

    if not np.isfinite(score_matrix).all():
        raise ValueError("The decision scores contain nonfinite values.")

    probability_bound_tolerance = 1e-12

    if np.any(probabilities < -probability_bound_tolerance):
        raise ValueError("The classifier returned negative probabilities.")

    if np.any(probabilities > 1.0 + probability_bound_tolerance):
        raise ValueError("The classifier returned probabilities greater than one.")

    probability_sums = probabilities.sum(axis=1)

    maximum_probability_sum_error = float(np.max(np.abs(probability_sums - 1.0)))

    if maximum_probability_sum_error > 1e-10:
        raise ValueError(
            "Predicted probabilities do not sum to one within "
            "the required tolerance. Maximum error: "
            f"{maximum_probability_sum_error:.17g}."
        )

    results = original_scenarios.copy()

    if "scenario_id" not in results.columns:
        results.insert(
            0,
            "scenario_id",
            np.arange(
                1,
                len(results) + 1,
                dtype=np.int64,
            ),
        )

    score_column_names: list[str] = []
    probability_column_names: list[str] = []

    for class_index, class_name in enumerate(classes):
        column_name = f"score_{class_name}"

        score_column_names.append(column_name)

        results[column_name] = score_matrix[:, class_index]

    for class_index, class_name in enumerate(classes):
        column_name = f"probability_{class_name}"

        probability_column_names.append(column_name)

        results[column_name] = probabilities[:, class_index]

    probability_order = np.argsort(
        probabilities,
        axis=1,
    )

    predicted_class_indices = probability_order[:, -1]

    second_class_indices = probability_order[:, -2]

    maximum_probabilities = probabilities[
        np.arange(len(probabilities)),
        predicted_class_indices,
    ]

    second_probabilities = probabilities[
        np.arange(len(probabilities)),
        second_class_indices,
    ]

    second_classes = np.asarray(
        classes,
        dtype=object,
    )[second_class_indices]

    confidence_margin = maximum_probabilities - second_probabilities

    safe_probabilities = np.clip(
        probabilities,
        np.finfo(np.float64).tiny,
        1.0,
    )

    prediction_entropy = -np.sum(
        safe_probabilities * np.log(safe_probabilities),
        axis=1,
    )

    normalized_entropy = prediction_entropy / np.log(len(classes))

    results["predicted_class"] = [str(value) for value in predicted_classes]

    results["maximum_probability"] = maximum_probabilities

    results["second_highest_class"] = [str(value) for value in second_classes]

    results["second_highest_probability"] = second_probabilities

    results["confidence_margin"] = confidence_margin

    results["prediction_entropy"] = prediction_entropy

    results["normalized_entropy"] = normalized_entropy

    results["probability_sum"] = probability_sums

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        float_format="%.17g",
    )

    terminal_columns = [
        "scenario_id",
        *score_column_names,
        *probability_column_names,
        "predicted_class",
        "maximum_probability",
        "second_highest_class",
        "second_highest_probability",
        "confidence_margin",
        "prediction_entropy",
        "normalized_entropy",
        "probability_sum",
    ]

    print()
    print("Prediction results")
    print("==================")

    print(
        results.loc[
            :,
            terminal_columns,
        ].to_string(
            index=False,
            max_rows=None,
            max_cols=None,
            float_format=lambda value: f"{value:.8g}",
        )
    )

    print()
    print("Prediction summary")
    print("==================")
    print(f"Model: {model_path.resolve()}")
    print(f"Input: {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Scenarios processed: {len(results)}")
    print(f"Predictor count: {len(required_columns)}")
    print(f"Extra columns preserved: {len(extra_columns)}")
    print(f"Target class count: {len(classes)}")
    print(f"Target classes: {classes}")
    print(f"Maximum probability-sum error: {maximum_probability_sum_error:.17g}")

    print()
    print("Predicted class counts")
    print("======================")

    print(
        results["predicted_class"]
        .value_counts()
        .reindex(
            classes,
            fill_value=0,
        )
        .to_string()
    )

    print()
    print(f"Results written to: {output_path.resolve()}")


def command_transform(args) -> None:
    """Export the fitted GAM design matrix for input scenarios."""

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Transformation input file does not exist: {input_path}"
        )

    model = joblib.load(model_path)

    if not hasattr(model, "named_steps"):
        raise ValueError("The loaded model is not a compatible fitted pipeline.")

    if "features" not in model.named_steps:
        raise ValueError("The loaded pipeline does not contain a 'features' step.")

    transformer = model.named_steps["features"]

    if not hasattr(
        transformer,
        "feature_names_in_",
    ):
        raise ValueError(
            "The fitted feature transformer does not expose feature_names_in_."
        )

    if not hasattr(
        transformer,
        "get_feature_names_out",
    ):
        raise ValueError(
            "The fitted feature transformer does not support get_feature_names_out()."
        )

    scenarios = pd.read_csv(input_path)

    if scenarios.empty:
        raise ValueError("The transformation input file contains no observations.")

    if scenarios.columns.has_duplicates:
        duplicate_columns = (
            scenarios.columns[scenarios.columns.duplicated()].astype(str).tolist()
        )

        raise ValueError(
            f"Transformation data contains duplicate column names: {duplicate_columns}."
        )

    required_columns = [str(value) for value in transformer.feature_names_in_]

    required_column_set = set(required_columns)

    input_column_set = {str(value) for value in scenarios.columns}

    missing_columns = sorted(required_column_set - input_column_set)

    if missing_columns:
        raise ValueError(
            f"Transformation data is missing required predictors: {missing_columns}."
        )

    extra_columns = [
        str(column)
        for column in scenarios.columns
        if str(column) not in required_column_set
    ]

    if extra_columns:
        print(
            "Extra input columns will be preserved in the output "
            "but excluded from GAM transformation: "
            f"{extra_columns}",
            flush=True,
        )

    model_scenarios = scenarios.loc[
        :,
        required_columns,
    ].copy()

    transformed_matrix = np.asarray(
        transformer.transform(model_scenarios),
        dtype=np.float64,
    )

    transformed_names = [str(value) for value in transformer.get_feature_names_out()]

    expected_shape = (
        len(model_scenarios),
        len(transformed_names),
    )

    if transformed_matrix.shape != expected_shape:
        raise ValueError(
            "The transformed GAM matrix has an unexpected shape. "
            f"Received {transformed_matrix.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(transformed_matrix).all():
        raise ValueError("The transformed GAM matrix contains nonfinite values.")

    if len(transformed_names) != len(set(transformed_names)):
        duplicated_names = sorted(
            {name for name in transformed_names if transformed_names.count(name) > 1}
        )

        raise ValueError(
            f"The transformed GAM feature names are not unique: {duplicated_names}."
        )

    transformed = pd.DataFrame(
        transformed_matrix,
        columns=transformed_names,
        index=scenarios.index,
    )

    metadata_columns = [
        column for column in scenarios.columns if str(column) not in required_column_set
    ]

    results = scenarios.loc[
        :,
        metadata_columns,
    ].copy()

    if "scenario_id" not in results.columns:
        results.insert(
            0,
            "scenario_id",
            np.arange(
                1,
                len(results) + 1,
                dtype=np.int64,
            ),
        )

    results = pd.concat(
        [
            results.reset_index(drop=True),
            transformed.reset_index(drop=True),
        ],
        axis=1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        float_format="%.17g",
    )

    prefix_counts = {
        "smooth_main_effects": sum(
            name.startswith("main_spline__") for name in transformed_names
        ),
        "linear_main_effects": sum(
            name.startswith("main_linear__") for name in transformed_names
        ),
        "categorical_main_effects": sum(
            name.startswith("main_categorical__") for name in transformed_names
        ),
        "interaction_components": sum(
            name.startswith("interaction__") for name in transformed_names
        ),
    }

    terminal_columns = [
        "scenario_id",
        *metadata_columns,
        *transformed_names,
    ]

    print()
    print("Transformed GAM components")
    print("==========================")

    print(
        results.loc[
            :,
            terminal_columns,
        ].to_string(
            index=False,
            max_rows=20,
            max_cols=20,
            float_format=lambda value: f"{value:.8g}",
        )
    )

    if len(results) > 20:
        print(f"\nTerminal preview limited to 20 of {len(results)} scenarios.")

    if len(terminal_columns) > 20:
        print(
            "Terminal preview limited to 20 columns. "
            "The CSV contains every transformed component."
        )

    print()
    print("Transformation summary")
    print("======================")
    print(f"Model: {model_path.resolve()}")
    print(f"Input: {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Scenarios transformed: {len(results)}")
    print(f"Input predictor count: {len(required_columns)}")
    print(f"Metadata columns preserved: {len(metadata_columns)}")
    print(f"Transformed component count: {len(transformed_names)}")
    print(f"Smooth main-effect components: {prefix_counts['smooth_main_effects']}")
    print(f"Linear main-effect components: {prefix_counts['linear_main_effects']}")
    print(
        "Categorical main-effect components: "
        f"{prefix_counts['categorical_main_effects']}"
    )
    print(f"Interaction components: {prefix_counts['interaction_components']}")

    print()
    print(f"Results written to: {output_path.resolve()}")


def command_contributions(args) -> None:
    """Export transformed-component contributions to class scores."""

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)
    top = int(args.top)

    if top < 1:
        raise ValueError("--top must be at least 1.")

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    if not input_path.is_file():
        raise FileNotFoundError(f"Contribution input file does not exist: {input_path}")

    model = joblib.load(model_path)

    if not hasattr(model, "named_steps"):
        raise ValueError("The loaded model is not a compatible fitted pipeline.")

    if "features" not in model.named_steps:
        raise ValueError("The loaded pipeline does not contain a 'features' step.")

    if "classifier" not in model.named_steps:
        raise ValueError("The loaded pipeline does not contain a 'classifier' step.")

    transformer = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    if not hasattr(
        transformer,
        "feature_names_in_",
    ):
        raise ValueError(
            "The fitted feature transformer does not expose feature_names_in_."
        )

    if not hasattr(
        transformer,
        "get_feature_names_out",
    ):
        raise ValueError(
            "The fitted feature transformer does not expose get_feature_names_out()."
        )

    if not hasattr(classifier, "classes_"):
        raise ValueError("The fitted classifier does not expose target classes.")

    scenarios = pd.read_csv(input_path)

    if scenarios.empty:
        raise ValueError("The contribution input file contains no observations.")

    if scenarios.columns.has_duplicates:
        duplicate_columns = (
            scenarios.columns[scenarios.columns.duplicated()].astype(str).tolist()
        )

        raise ValueError(
            f"Contribution data contains duplicate column names: {duplicate_columns}."
        )

    required_columns = [str(value) for value in transformer.feature_names_in_]

    required_column_set = set(required_columns)

    input_column_set = {str(value) for value in scenarios.columns}

    missing_columns = sorted(required_column_set - input_column_set)

    if missing_columns:
        raise ValueError(
            f"Contribution data is missing required predictors: {missing_columns}."
        )

    metadata_columns = [
        str(column)
        for column in scenarios.columns
        if str(column) not in required_column_set
    ]

    if metadata_columns:
        print(
            "Extra input columns will be preserved as scenario "
            "metadata but excluded from model calculation: "
            f"{metadata_columns}",
            flush=True,
        )

    model_scenarios = scenarios.loc[
        :,
        required_columns,
    ].copy()

    transformed_matrix = np.asarray(
        transformer.transform(model_scenarios),
        dtype=np.float64,
    )

    transformed_names = [str(value) for value in transformer.get_feature_names_out()]

    if len(transformed_names) != len(set(transformed_names)):
        duplicate_names = sorted(
            {name for name in transformed_names if transformed_names.count(name) > 1}
        )

        raise ValueError(
            f"The transformed GAM feature names are not unique: {duplicate_names}."
        )

    expected_transformed_shape = (
        len(model_scenarios),
        len(transformed_names),
    )

    if transformed_matrix.shape != expected_transformed_shape:
        raise ValueError(
            "The transformed GAM matrix has an unexpected shape. "
            f"Received {transformed_matrix.shape}; "
            f"expected {expected_transformed_shape}."
        )

    if not np.isfinite(transformed_matrix).all():
        raise ValueError("The transformed GAM matrix contains nonfinite values.")

    parameters = extract_class_score_parameters(classifier)

    classes = [str(value) for value in parameters.classes]

    if len(classes) < 2:
        raise ValueError("The classifier must contain at least two classes.")

    coefficients = np.asarray(
        parameters.coefficients,
        dtype=np.float64,
    )

    intercepts = np.asarray(
        parameters.intercepts,
        dtype=np.float64,
    )

    expected_coefficient_shape = (
        len(classes),
        len(transformed_names),
    )

    if coefficients.shape != expected_coefficient_shape:
        raise ValueError(
            "The coefficient matrix has an unexpected shape. "
            f"Received {coefficients.shape}; "
            f"expected {expected_coefficient_shape}."
        )

    expected_intercept_shape = (len(classes),)

    if intercepts.shape != expected_intercept_shape:
        raise ValueError(
            "The intercept array has an unexpected shape. "
            f"Received {intercepts.shape}; "
            f"expected {expected_intercept_shape}."
        )

    if not np.isfinite(coefficients).all():
        raise ValueError("The coefficient matrix contains nonfinite values.")

    if not np.isfinite(intercepts).all():
        raise ValueError("The intercept array contains nonfinite values.")

    probabilities = np.asarray(
        model.predict_proba(model_scenarios),
        dtype=np.float64,
    )

    predicted_classes = np.asarray(
        model.predict(model_scenarios),
        dtype=object,
    )

    expected_output_shape = (
        len(model_scenarios),
        len(classes),
    )

    if probabilities.shape != expected_output_shape:
        raise ValueError(
            "The probability matrix has an unexpected shape. "
            f"Received {probabilities.shape}; "
            f"expected {expected_output_shape}."
        )

    expected_prediction_shape = (len(model_scenarios),)

    if predicted_classes.shape != expected_prediction_shape:
        raise ValueError(
            "The predicted-class array has an unexpected shape. "
            f"Received {predicted_classes.shape}; "
            f"expected {expected_prediction_shape}."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError("The predicted probabilities contain nonfinite values.")

    score_matrix = transformed_matrix @ coefficients.T + intercepts

    if score_matrix.shape != expected_output_shape:
        raise ValueError(
            "The reconstructed class-score matrix has an "
            "unexpected shape. "
            f"Received {score_matrix.shape}; "
            f"expected {expected_output_shape}."
        )

    if not np.isfinite(score_matrix).all():
        raise ValueError("The reconstructed class scores contain nonfinite values.")

    raw_scores = np.asarray(
        model.decision_function(model_scenarios),
        dtype=np.float64,
    )

    if raw_scores.ndim == 1:
        if len(classes) != 2:
            raise ValueError(
                "One-dimensional decision scores are valid only "
                "for binary classification."
            )

        expected_scores = np.column_stack(
            [
                np.zeros_like(raw_scores),
                raw_scores,
            ]
        )

    elif raw_scores.ndim == 2:
        expected_scores = raw_scores

    else:
        raise ValueError(
            "The classifier returned decision scores with an "
            f"unsupported shape: {raw_scores.shape}."
        )

    if expected_scores.shape != expected_output_shape:
        raise ValueError(
            "The fitted classifier score matrix has an unexpected "
            "shape. "
            f"Received {expected_scores.shape}; "
            f"expected {expected_output_shape}."
        )

    maximum_score_error = float(np.max(np.abs(score_matrix - expected_scores)))

    score_tolerance = 1e-10

    if maximum_score_error > score_tolerance:
        raise ValueError(
            "Reconstructed class scores do not match the fitted "
            "classifier. Maximum error: "
            f"{maximum_score_error:.17g}."
        )

    if "scenario_id" in scenarios.columns:
        scenario_ids = scenarios["scenario_id"].copy()
    else:
        scenario_ids = pd.Series(
            np.arange(
                1,
                len(scenarios) + 1,
                dtype=np.int64,
            ),
            index=scenarios.index,
            name="scenario_id",
        )

    metadata_frame = scenarios.loc[
        :,
        metadata_columns,
    ].copy()

    if "scenario_id" not in metadata_frame.columns:
        metadata_frame.insert(
            0,
            "scenario_id",
            scenario_ids.to_numpy(),
        )

    rows: list[dict[str, object]] = []

    for observation_index in range(len(model_scenarios)):
        scenario_id = scenario_ids.iloc[observation_index]

        predicted_class = str(predicted_classes[observation_index])

        transformed_values = transformed_matrix[observation_index]

        for class_index, class_name in enumerate(classes):
            class_probability = float(
                probabilities[
                    observation_index,
                    class_index,
                ]
            )

            class_score = float(
                score_matrix[
                    observation_index,
                    class_index,
                ]
            )

            intercept = float(intercepts[class_index])

            rows.append(
                {
                    "scenario_id": scenario_id,
                    "observation_index": (observation_index),
                    "class": class_name,
                    "predicted_class": predicted_class,
                    "class_probability": class_probability,
                    "class_score": class_score,
                    "component": "intercept",
                    "component_type": "intercept",
                    "component_group": "intercept",
                    "transformed_value": 1.0,
                    "coefficient": intercept,
                    "contribution": intercept,
                    "absolute_contribution": abs(intercept),
                }
            )

            class_coefficients = coefficients[class_index]

            contributions = transformed_values * class_coefficients

            for (
                transformed_name,
                transformed_value,
                coefficient,
                contribution,
            ) in zip(
                transformed_names,
                transformed_values,
                class_coefficients,
                contributions,
                strict=True,
            ):
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "observation_index": (observation_index),
                        "class": class_name,
                        "predicted_class": (predicted_class),
                        "class_probability": (class_probability),
                        "class_score": class_score,
                        "component": transformed_name,
                        "component_type": (_component_type(transformed_name)),
                        "component_group": (_component_group(transformed_name)),
                        "transformed_value": float(transformed_value),
                        "coefficient": float(coefficient),
                        "contribution": float(contribution),
                        "absolute_contribution": float(abs(contribution)),
                    }
                )

    results = pd.DataFrame(
        rows,
        columns=[
            "scenario_id",
            "observation_index",
            "class",
            "predicted_class",
            "class_probability",
            "class_score",
            "component",
            "component_type",
            "component_group",
            "transformed_value",
            "coefficient",
            "contribution",
            "absolute_contribution",
        ],
    )

    contribution_sums = (
        results.groupby(
            [
                "scenario_id",
                "class",
            ],
            sort=False,
        )["contribution"]
        .sum()
        .rename("reconstructed_score")
        .reset_index()
    )

    score_summary_rows: list[dict[str, object]] = []

    for observation_index in range(len(model_scenarios)):
        scenario_id = scenario_ids.iloc[observation_index]

        predicted_class = str(predicted_classes[observation_index])

        for class_index, class_name in enumerate(classes):
            score_summary_rows.append(
                {
                    "scenario_id": scenario_id,
                    "observation_index": (observation_index),
                    "class": class_name,
                    "predicted_class": (predicted_class),
                    "class_probability": float(
                        probabilities[
                            observation_index,
                            class_index,
                        ]
                    ),
                    "expected_score": float(
                        expected_scores[
                            observation_index,
                            class_index,
                        ]
                    ),
                }
            )

    score_summary = pd.DataFrame(score_summary_rows)

    contribution_sums = contribution_sums.merge(
        score_summary,
        on=[
            "scenario_id",
            "class",
        ],
        validate="one_to_one",
    )

    contribution_sums["absolute_score_error"] = np.abs(
        contribution_sums["reconstructed_score"] - contribution_sums["expected_score"]
    )

    maximum_contribution_sum_error = float(
        contribution_sums["absolute_score_error"].max()
    )

    if maximum_contribution_sum_error > score_tolerance:
        raise ValueError(
            "Component contributions do not sum to the class "
            "scores within the required tolerance. Maximum "
            "error: "
            f"{maximum_contribution_sum_error:.17g}."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        float_format="%.17g",
    )

    summary_path = output_path.with_name(f"{output_path.stem}_score_summary.csv")

    contribution_sums.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
        float_format="%.17g",
    )

    if metadata_columns:
        metadata_path = output_path.with_name(
            f"{output_path.stem}_scenario_metadata.csv"
        )

        metadata_frame.to_csv(
            metadata_path,
            index=False,
            encoding="utf-8",
        )
    else:
        metadata_path = None

    terminal_results = (
        results.sort_values(
            [
                "observation_index",
                "class",
                "absolute_contribution",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .groupby(
            [
                "observation_index",
                "class",
            ],
            sort=False,
            group_keys=False,
        )
        .head(top)
    )

    terminal_columns = [
        "scenario_id",
        "class",
        "predicted_class",
        "class_probability",
        "class_score",
        "component",
        "transformed_value",
        "coefficient",
        "contribution",
        "absolute_contribution",
    ]

    print()
    print("Largest class-score contributions")
    print("=================================")

    print(
        terminal_results.loc[
            :,
            terminal_columns,
        ].to_string(
            index=False,
            max_rows=None,
            max_cols=None,
            float_format=lambda value: f"{value:.8g}",
        )
    )

    print()
    print("Contribution summary")
    print("====================")
    print(f"Model: {model_path.resolve()}")
    print(f"Input: {input_path.resolve()}")
    print(f"Output: {output_path.resolve()}")
    print(f"Score summary: {summary_path.resolve()}")

    if metadata_path is not None:
        print(f"Scenario metadata: {metadata_path.resolve()}")

    print(f"Scenarios processed: {len(model_scenarios)}")
    print(f"Target class count: {len(classes)}")
    print(f"Target classes: {classes}")
    print(f"Transformed component count: {len(transformed_names)}")
    print(f"Contribution rows written: {len(results)}")
    print(f"Maximum class-score reconstruction error: {maximum_score_error:.17g}")
    print(f"Maximum contribution-sum error: {maximum_contribution_sum_error:.17g}")


def command_demo(args) -> None:
    rng = np.random.default_rng(args.seed)
    n = args.rows
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-2, 2, size=n)
    x3 = rng.choice(["low", "standard", "high"], size=n)
    x4 = rng.normal(size=n)
    scores = np.column_stack(
        [
            1.2 * np.sin(x1) - 0.5 * x2,
            -0.8 * x1 + 0.7 * x2**2,
            0.6 * x1 * x2 + 0.4 * x4,
            -0.4 * x2 - 0.5 * x4,
        ]
    )
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = np.array(["A", "B", "C", "D"])
    y = np.array([rng.choice(labels, p=row) for row in probabilities])
    frame = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "X4": x4, "Y": y})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Demonstration dataset written to {args.output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gam-app")
    sub = parser.add_subparsers(dest="command", required=True)
    profile = sub.add_parser("profile")
    profile.add_argument("--data", type=Path, required=True)
    profile.add_argument("--target", required=True)
    profile.add_argument("--output", type=Path, required=True)
    profile.set_defaults(func=command_profile)
    configure = sub.add_parser("configure")
    configure.add_argument("--data", type=Path, required=True)
    configure.add_argument("--target", required=True)
    configure.add_argument("--output", type=Path, required=True)
    configure.add_argument("--name")
    configure.add_argument("--row-id")
    configure.add_argument(
        "--preset", choices=["quick", "standard", "thorough"], default="standard"
    )
    configure.add_argument("--non-interactive", action="store_true")
    configure.set_defaults(func=command_configure)
    plan = sub.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.set_defaults(func=command_plan)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--workspace", type=Path, default=Path("workspace"))
    run.set_defaults(func=command_run)
    status = sub.add_parser("status")
    status.add_argument("--run", type=Path, required=True)
    status.add_argument("--follow", action="store_true")
    status.set_defaults(func=command_status)
    resume = sub.add_parser("resume")
    resume.add_argument("--run", type=Path, required=True)
    resume.set_defaults(func=command_resume)
    for name, marker in [("pause", "PAUSE"), ("cancel", "CANCEL")]:
        command = sub.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.set_defaults(func=lambda args, m=marker: command_control(args, m))
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--run", type=Path, required=True)
    inspect.add_argument("--model", required=True)
    inspect.add_argument(
        "--reference-class",
        default=None,
        help=(
            "Reference target class for contrast equations. "
            "Defaults to the classifier's first class."
        ),
    )
    inspect.set_defaults(func=command_inspect)
    link = sub.add_parser("verify-link")
    link.add_argument("--run", type=Path, required=True)
    link.add_argument("--model", required=True)
    link.set_defaults(func=command_verify_link)
    compare = sub.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--left-model", required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--right-model", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(func=command_compare)
    predict = sub.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.set_defaults(func=command_predict)

    transform_parser = sub.add_parser(
        "transform",
        help=("Export the fitted GAM design matrix for predictor scenarios."),
    )

    transform_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a fitted model.joblib file.",
    )

    transform_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV file containing predictor scenarios.",
    )

    transform_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV for transformed GAM components.",
    )

    transform_parser.set_defaults(
        func=command_transform,
    )

    contributions_parser = sub.add_parser(
        "contributions",
        help=("Export transformed-component contributions to every class score."),
    )

    contributions_parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a fitted model.joblib file.",
    )

    contributions_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV file containing predictor scenarios.",
    )

    contributions_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV for component contributions.",
    )

    contributions_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help=(
            "Number of largest absolute contributions to display "
            "per scenario and class. The output CSV always contains "
            "all components."
        ),
    )

    contributions_parser.set_defaults(
        func=command_contributions,
    )

    demo = sub.add_parser("demo")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--rows", type=int, default=300)
    demo.add_argument("--seed", type=int, default=42)
    demo.set_defaults(func=command_demo)
    return parser


def main() -> None:
    try:
        args = build_parser().parse_args()
        args.func(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
