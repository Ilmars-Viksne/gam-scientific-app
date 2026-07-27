from __future__ import annotations

import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .config import ExperimentConfig
from .run_store import FileRunStore


def create_reports(config: ExperimentConfig, store: FileRunStore) -> None:
    sections = [f"<h1>{html.escape(config.name)}</h1>"]
    sections.append("<p>Penalized multinomial logistic additive B-spline experiment.</p>")
    for model in config.models:
        result_dir = store.results / model.id
        summary = pd.read_csv(result_dir / "summary.csv", index_col=0)
        predictions = pd.read_parquet(result_dir / "predictions.parquet")
        sections.append(f"<h2>{html.escape(model.id)}</h2>")
        sections.append(summary.to_html(float_format=lambda value: f"{value:.6f}"))
        labels = sorted(predictions["observed_class"].unique())
        matrix = confusion_matrix(predictions.observed_class, predictions.predicted_class, labels=labels)
        display = ConfusionMatrixDisplay(matrix, display_labels=labels)
        display.plot()
        path = store.plots / f"{model.id}_confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        sections.append(f'<img src="../plots/{path.name}" alt="Confusion matrix">')
        sections.append(f'<p><a href="../results/{model.id}/fold_metrics.csv">Fold metrics</a></p>')
    report = "<!doctype html><html><head><meta charset='utf-8'><title>GAM report</title>"
    report += "<style>body{font-family:Arial;max-width:1100px;margin:2rem auto}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:.4rem}img{max-width:700px}</style>"
    report += "</head><body>" + "\n".join(sections) + "</body></html>"
    (store.reports / "report.html").write_text(report, encoding="utf-8")
