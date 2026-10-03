"""Check scientific transformations and headless artifact-only figure exports."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from creditrisk.visualization import (
    FIGURE_STEMS, PARAMETER_BOUNDS, confusion_matrix_from_row,
    generate_chapter3_figures, normalize_hyperparameters, paired_metric_table,
    plot_final_test_performance, prepare_tradeoff_landscape,
    shap_rank_matrix, summarize_shap_importance,
)

ROOT = Path(__file__).resolve().parents[1]


def test_parameter_normalization_uses_locked_bounds_and_log_midpoints():
    rows = []
    for position in (0.0, 0.5, 1.0):
        row = {}
        for name, (low, high, log) in PARAMETER_BOUNDS.items():
            value = low * (high / low)**position if log else low + position*(high-low)
            row[f"param_{name}"] = value
        rows.append(row)
    actual = normalize_hyperparameters(pd.DataFrame(rows))
    np.testing.assert_allclose(actual.iloc[0], 0, atol=1e-12)
    np.testing.assert_allclose(actual.iloc[1], 0.5, atol=1e-12)
    np.testing.assert_allclose(actual.iloc[2], 1, atol=1e-12)
    # Removing other configurations must not change any coordinate.
    np.testing.assert_allclose(normalize_hyperparameters(pd.DataFrame(rows[1:2])).iloc[0], actual.iloc[1])


def test_parameter_normalization_rejects_out_of_protocol_value():
    table = pd.DataFrame([{f"param_{name}": low for name, (low, _, _) in PARAMETER_BOUNDS.items()}])
    table["param_subsample"] = 0.4
    with pytest.raises(ValueError, match="outside"):
        normalize_hyperparameters(table)


def test_shap_summary_keeps_observed_scale_population_sd_and_tie_ranks():
    importance = pd.DataFrame({"fold": [0, 1], "A": [4., 2.], "B": [2., 2.], "C": [0., 6.]})
    summary = summarize_shap_importance(importance).set_index("feature")
    assert summary.loc["A", "mean"] == 3.0
    assert summary.loc["A", "std"] == 1.0
    assert summary.loc["C", "std"] == 3.0
    ranks = shap_rank_matrix(importance)
    assert ranks.loc["A", 0] == 1.0
    assert ranks.loc["C", 1] == 1.0
    assert ranks.loc["A", 1] == ranks.loc["B", 1] == 2.5


def test_shap_column_padding_is_normalized_without_mutating_input():
    table = pd.DataFrame({" fold ": [0, 1], " A ": [4., 2.], " B ": [2., 4.]})
    original = table.copy(deep=True)
    summary = summarize_shap_importance(table)
    assert set(summary["feature"]) == {"A", "B"}
    assert list(shap_rank_matrix(table).columns) == [0, 1]
    pd.testing.assert_frame_equal(table, original)
    with pytest.raises(ValueError, match="unique"):
        summarize_shap_importance(table.assign(A=[1., 2.]))


@pytest.mark.parametrize("importance", [
    pd.DataFrame({"fold": [0, 0], "A": [1., 2.]}),
    pd.DataFrame({"fold": [0, 1], "A": [1., np.nan]}),
    pd.DataFrame({"fold": [0, 1], "A": [1., -1.]}),
])
def test_shap_summary_rejects_invalid_reference_artifacts(importance):
    with pytest.raises(ValueError):
        summarize_shap_importance(importance)


def test_paired_partition_values_align_by_seed_not_input_order():
    repeats = pd.DataFrame({"seed": [102, 101, 101, 102],
                            "candidate": ["tpe_best", "pareto_balanced", "tpe_best", "pareto_balanced"],
                            "shap_stability": [0.8, 0.95, 0.9, 0.94]})
    wide = paired_metric_table(repeats, "shap_stability")
    np.testing.assert_allclose(wide["pareto_balanced"] - wide["tpe_best"], [0.05, 0.14])
    with pytest.raises(ValueError, match="complete set"):
        paired_metric_table(repeats.iloc[:-1], "shap_stability")
    with pytest.raises(ValueError, match="Duplicate"):
        paired_metric_table(pd.concat([repeats, repeats.iloc[:1]]), "shap_stability")


def test_confusion_matrix_preserves_label_order_and_observed_recall():
    row = {"tn": 4437, "fp": 236, "fn": 859, "tp": 468,
           "n_test": 6000, "n_positive": 1327, "n_negative": 4673}
    matrix = confusion_matrix_from_row(row)
    np.testing.assert_array_equal(matrix, [[4437, 236], [859, 468]])
    assert matrix[1, 1] / matrix[1].sum() == pytest.approx(0.35267520723436324)
    with pytest.raises(ValueError, match="sum"):
        confusion_matrix_from_row({**row, "n_test": 5999})
    with pytest.raises(ValueError, match="integers"):
        confusion_matrix_from_row({**row, "tp": 468.5})


def test_landscape_joins_configurations_without_inventing_test_stability():
    comparison = pd.read_csv(ROOT / "artifacts/tables/xgb_pareto_comparison.csv")
    final = pd.read_csv(ROOT / "artifacts/tables/xgb_final_test_summary.csv")
    landscape = prepare_tradeoff_landscape(comparison.sample(frac=1, random_state=1), final.iloc[::-1])
    balanced = landscape.set_index("candidate").loc["pareto_balanced"]
    assert balanced["roc_auc_test"] == pytest.approx(0.781790839034096)
    assert balanced["shap_stability"] == pytest.approx(0.9501976284584981)
    assert not any("test" in column and "stability" in column for column in landscape)
    with pytest.raises(ValueError, match="do not match"):
        prepare_tradeoff_landscape(comparison, final.iloc[:-1])


def test_final_plot_uses_persisted_counts_and_reports_unavailable_curves():
    summary = pd.read_csv(ROOT / "artifacts/tables/xgb_final_test_summary.csv")
    fig = plot_final_test_performance(summary)
    try:
        assert len(fig.axes) == 7
        np.testing.assert_array_equal(fig.axes[5].images[0].get_array(), [[4437, 236], [859, 468]])
        assert "ROC/PR curves unavailable" in " ".join(text.get_text() for text in fig.texts)
        assert not any(ax.lines for ax in fig.axes[:2])
    finally:
        plt.close(fig)


def test_complete_pipeline_exports_valid_png300_and_vector_pdf_with_provenance(tmp_path, monkeypatch):
    # Guard reads as well as output validity: a figure-only run must use artifacts,
    # never raw/Core/Test observations or the Optuna database.
    read_csv = pd.read_csv
    reads = []

    def artifacts_only(path, *args, **kwargs):
        path = Path(path).resolve()
        assert ROOT / "artifacts" in path.parents
        assert path.suffix == ".csv"
        reads.append(path)
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", artifacts_only)
    final_manifest_path = ROOT / "artifacts/tables/xgb_final_test_manifest.json"
    final_manifest_before = final_manifest_path.read_bytes()
    result = generate_chapter3_figures(ROOT, tmp_path)
    assert len(result["figures"]) == 2 * len(FIGURE_STEMS)
    assert len(reads) >= 9
    assert result["test_re_evaluated"] is False
    assert result["protocol"] == "artifact_only_no_fitting_no_test_inference"
    assert final_manifest_path.read_bytes() == final_manifest_before
    stored = json.loads((tmp_path / "figure_manifest.json").read_text())
    assert stored == result
    for artifact in result["figures"]:
        path = tmp_path / artifact["file"]
        assert path.stat().st_size > 10_000
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        if path.suffix == ".png":
            with Image.open(path) as image:
                assert image.format == "PNG"
                assert min(image.size) > 1000
                assert image.info["dpi"] == pytest.approx((300, 300), abs=0.01)
                image.verify()
        else:
            assert path.read_bytes().startswith(b"%PDF-")
            assert b"/Font" in path.read_bytes()
    assert "Unavailable" in result["limitations"]["final_test_roc_pr"]
    assert not plt.get_fignums()
