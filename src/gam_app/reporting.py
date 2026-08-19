from __future__ import annotations

import html
import json

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .config import ExperimentConfig
from .run_store import FileRunStore


def create_reports(config: ExperimentConfig, store: FileRunStore) -> None:
    sections = [f"<h1>{html.escape(config.name)}</h1>"]
    sections.append(
        "<p>Penalized multinomial logistic additive B-spline experiment.</p>"
    )

    data_manifest_path = store.root / "data_manifest.json"
    if data_manifest_path.exists():
        data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        rows = data_manifest.get("rows", 0)
        predictors = data_manifest.get("predictors", [])
        target = data_manifest.get("target", "")
        class_counts = data_manifest.get("class_counts", {})

        dist_rows = []
        for cls_name, count in sorted(class_counts.items(), key=lambda x: str(x[0])):
            pct = (count / rows * 100.0) if rows > 0 else 0.0
            dist_rows.append(
                {
                    "class": str(cls_name),
                    "count": count,
                    "percentage": f"{pct:.2f}%",
                }
            )
        dist_df = pd.DataFrame(dist_rows)

        sections.append("<h2>Dataset overview</h2>")
        sections.append(f"<p><strong>Number of observations:</strong> {rows}</p>")
        sections.append(
            f"<p><strong>Number of active predictors:</strong> {len(predictors)}</p>"
        )
        sections.append(
            f"<p><strong>Target:</strong> {html.escape(str(target))}</p>"
        )
        sections.append("<h3>Class distribution</h3>")
        sections.append(dist_df.to_html(index=False))

    for model in config.models:
        result_dir = store.results / model.id
        summary = pd.read_csv(result_dir / "summary.csv", index_col=0)
        predictions = pd.read_parquet(result_dir / "predictions.parquet")

        sections.append(f"<h2>{html.escape(model.id)}</h2>")

        metadata_path = store.models / model.id / "model_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            meta_df = pd.DataFrame(
                [
                    {"property": k, "value": str(v)}
                    for k, v in metadata.items()
                ]
            )
            sections.append("<h3>Final-model metadata</h3>")
            sections.append(meta_df.to_html(index=False))

        params_path = store.models / model.id / "best_parameters.json"
        if params_path.exists():
            params = json.loads(params_path.read_text(encoding="utf-8"))
            params_df = pd.DataFrame([params])
            sections.append(
                "<h3>Final full-data inner-CV hyperparameter selection</h3>"
            )
            sections.append(
                params_df.to_html(
                    index=False,
                    float_format=lambda value: f"{value:.6f}",
                )
            )

        sections.append("<h3>Nested CV metric summary</h3>")
        sections.append(
            summary.to_html(float_format=lambda value: f"{value:.6f}")
        )

        class_metrics_path = result_dir / "class_metrics.csv"
        if class_metrics_path.exists():
            class_metrics = pd.read_csv(class_metrics_path)
            class_summary = (
                class_metrics.groupby("class", as_index=False)
                .agg(
                    sensitivity_mean=("sensitivity", "mean"),
                    specificity_mean=("specificity", "mean"),
                    precision_mean=("precision", "mean"),
                    f1_mean=("f1", "mean"),
                    mean_fold_support=("support", "mean"),
                    total_oof_support=("support", "sum"),
                )
            )
            class_summary["total_oof_support"] = class_summary[
                "total_oof_support"
            ].astype(int)
            sections.append("<h3>Per-class performance summary</h3>")
            sections.append(
                class_summary.to_html(
                    index=False,
                    float_format=lambda value: f"{value:.6f}",
                )
            )

        labels = sorted(predictions["observed_class"].unique())

        raw_matrix = confusion_matrix(
            predictions.observed_class, predictions.predicted_class, labels=labels
        )
        display = ConfusionMatrixDisplay(raw_matrix, display_labels=labels)
        display.plot()
        raw_path = store.plots / f"{model.id}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(raw_path, dpi=160)
        plt.close()

        norm_matrix = confusion_matrix(
            predictions.observed_class,
            predictions.predicted_class,
            labels=labels,
            normalize="true",
        )
        norm_display = ConfusionMatrixDisplay(norm_matrix, display_labels=labels)
        norm_display.plot(values_format=".3f")
        norm_path = store.plots / f"{model.id}_confusion_matrix_normalized.png"
        plt.tight_layout()
        plt.savefig(norm_path, dpi=160)
        plt.close()

        sections.append("<h3>Confusion matrices</h3>")
        sections.append("<p><strong>Pooled raw count matrix:</strong></p>")
        sections.append(
            f'<img src="../plots/{raw_path.name}" alt="Confusion matrix">'
        )
        sections.append("<p><strong>Pooled row-normalized matrix:</strong></p>")
        sections.append(
            f'<img src="../plots/{norm_path.name}" alt="Normalized confusion matrix">'
        )

        sections.append(
            "<p><em>Note: These confusion matrices pool held-out outer-fold "
            "predictions across all outer repeats. Each observation contributes "
            "one out-of-fold prediction per repeat.</em></p>"
        )

        total_predictions = len(predictions)
        unique_obs = (
            predictions["row_id"].nunique()
            if "row_id" in predictions.columns
            else ""
        )
        outer_repeats = (
            predictions["repeat"].nunique()
            if "repeat" in predictions.columns
            else ""
        )

        count_info = []
        if unique_obs != "":
            count_info.append(f"Unique observations: {unique_obs}")
        if outer_repeats != "":
            count_info.append(f"Outer repeats: {outer_repeats}")
        count_info.append(f"Total OOF prediction events: {total_predictions}")

        sections.append(f"<p>{' | '.join(count_info)}</p>")

        sections.append("<h3>Detailed result files</h3>")
        mid = model.id
        links = [
            f'<li><a href="../results/{mid}/fold_metrics.csv">'
            "Detailed fold metrics</a></li>",
            f'<li><a href="../results/{mid}/class_metrics.csv">'
            "Detailed per-class fold metrics</a></li>",
            f'<li><a href="../results/{mid}/class_metrics_summary.csv">'
            "Per-class summary statistics</a></li>",
            f'<li><a href="../results/{mid}/predictions.parquet">'
            "Out-of-fold predictions</a></li>",
        ]
        sections.append(f"<ul>{''.join(links)}</ul>")

        inspection_dir = result_dir / "inspection"
        if inspection_dir.exists():
            sections.append("<h3>Inspection artifacts</h3>")
            insp_links = []
            if (inspection_dir / "equations.txt").exists():
                insp_links.append(
                    f'<li><a href="../results/{mid}/inspection/equations.txt">'
                    "View transformed-space equations</a></li>"
                )
            if (inspection_dir / "reference_equations.csv").exists():
                insp_links.append(
                    f'<li><a href="../results/{mid}/inspection/'
                    'reference_equations.csv">View reference-class contrasts</a></li>'
                )
            if (inspection_dir / "components.csv").exists():
                insp_links.append(
                    f'<li><a href="../results/{mid}/inspection/components.csv">'
                    "View coefficient components</a></li>"
                )
            if insp_links:
                sections.append(f"<ul>{''.join(insp_links)}</ul>")

        link_verification_dir = result_dir / "link_verification"
        if link_verification_dir.exists():
            sections.append("<h3>Link-verification results</h3>")
            ver_txt = link_verification_dir / "verification.txt"
            if ver_txt.exists():
                ver_content = ver_txt.read_text(encoding="utf-8")
                sections.append(f"<pre>{html.escape(ver_content)}</pre>")
            ver_links = []
            if (link_verification_dir / "scores.csv").exists():
                ver_links.append(
                    f'<li><a href="../results/{mid}/link_verification/scores.csv">'
                    "Scores</a></li>"
                )
            if (link_verification_dir / "probabilities.csv").exists():
                ver_links.append(
                    f'<li><a href="../results/{mid}/link_verification/'
                    'probabilities.csv">Probabilities</a></li>'
                )
            if (link_verification_dir / "reconstructed_probabilities.csv").exists():
                ver_links.append(
                    f'<li><a href="../results/{mid}/link_verification/'
                    'reconstructed_probabilities.csv">'
                    "Reconstructed probabilities</a></li>"
                )
            if ver_links:
                sections.append(f"<ul>{''.join(ver_links)}</ul>")

    report = (
        "<!doctype html><html><head><meta charset='utf-8'><title>GAM report</title>"
    )

    report += (
        "<style>"
        "body{font-family:Arial;max-width:1100px;margin:2rem auto}"
        "table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc;padding:.4rem}"
        "img{max-width:700px}"
        "</style>"
    )

    report += "</head><body>" + "\n".join(sections) + "</body></html>"
    (store.reports / "report.html").write_text(report, encoding="utf-8")
