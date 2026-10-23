"""Read-only baseline verification; writes only this audit's JSON result.

No imports from the ML runtime, model execution, package changes, or rewrites
to the saved baseline are performed. Package inventory is read via pip freeze.
"""
import hashlib
import json
import math
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit_stage1"
FROZEN = ROOT / "baseline_v0"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    manifest = json_read(FROZEN / "manifest.json")
    record = json_read(FROZEN / "record.json")
    results = []
    bytes_checked = 0
    for item in manifest:
        path = (ROOT / item["path"]).resolve()
        if ROOT.resolve() not in path.parents:
            results.append({"path": item["path"], "status": "UNSAFE_PATH"})
            continue
        if not path.is_file():
            results.append({"path": item["path"], "status": "MISSING"})
            continue
        size = path.stat().st_size
        actual = sha256(path)
        bytes_checked += size
        if size != item["bytes"] or actual != item["sha256"]:
            results.append({"path": item["path"], "status": "MISMATCH",
                            "actual_bytes": size, "actual_sha256": actual})
    unique_paths = len(set(item["path"] for item in manifest))
    files_pass = (not results and len(manifest) == record["file_count"]
                  and unique_paths == len(manifest))

    archive = FROZEN / record["snapshot"]
    archive_sha = sha256(archive)
    archive_bytes = archive.stat().st_size
    archive_pass = (archive_sha == record["snapshot_sha256"]
                    and archive_bytes == record["snapshot_bytes"])

    frozen_lines = sorted(line.strip() for line in
                          (FROZEN / "environment-freeze.txt").read_text(encoding="utf-8-sig").splitlines()
                          if line.strip())
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    run = subprocess.run([str(python), "-m", "pip", "freeze", "--all"],
                         capture_output=True, text=True, timeout=60, check=False)
    current_lines = sorted(line.strip() for line in run.stdout.splitlines() if line.strip())
    environment_pass = run.returncode == 0 and current_lines == frozen_lines

    metrics_path = AUDIT / "metrics_recheck.json"
    metrics = json_read(metrics_path)
    assertions = metrics["assertions"]
    inputs = {}
    for label, entry in metrics["inputs"].items():
        path = ROOT / entry["path"]
        digest = sha256(path)
        inputs[label] = {"sha256": digest, "matches_record": digest == entry["sha256"]}
    matrix = metrics["count_confusion_matrix"]
    supports = [sum(row) for row in matrix]
    prediction_counts = [sum(row[column] for row in matrix) for column in range(len(matrix))]
    total = sum(supports)
    top1 = sum(matrix[index][index] for index in range(len(matrix))) / total
    mean1 = sum(matrix[index][index] / supports[index] for index in range(len(matrix))) / len(matrix)
    samples = metrics["samples"]
    top5 = sum(sample["true_label"] in sample["top5_labels"] for sample in samples) / len(samples)
    losses = metrics["recorded_training"]["loss"]
    mean_loss = sum(losses) / len(losses)
    recomputed = {"acc/top1": top1, "acc/top5": top5, "acc/mean1": mean1}
    checks = {
        "recorded_28_assertions_pass": len(assertions) == 28 and all(a["status"] == "PASS" for a in assertions),
        "saved_inputs_hash_match": all(item["matches_record"] for item in inputs.values()),
        "ten_by_ten_count_matrix": len(matrix) == 10 and all(len(row) == 10 for row in matrix),
        "count_matrix_totals_10": total == 10,
        "class_support_matches": supports == metrics["class_support"] == [1] * 10,
        "prediction_counts_match": prediction_counts == metrics["prediction_counts"],
        "metrics_arithmetic_matches": all(math.isclose(value, metrics["recomputed_engineering_metrics"][key], rel_tol=0, abs_tol=1e-12) for key, value in recomputed.items()),
        "ten_finite_losses": len(losses) == 10 and all(math.isfinite(value) for value in losses),
        "mean_loss_matches": math.isclose(mean_loss, metrics["recorded_training"]["mean_loss"], rel_tol=0, abs_tol=1e-12),
        "validation_loss_unavailable_explicit": metrics["recorded_training"]["validation_loss_available"] is False,
    }
    figures = {}
    for name in ("baseline_v0_loss.png", "baseline_v0_loss.svg", "baseline_v0_confusion.png", "baseline_v0_confusion.svg"):
        path = AUDIT / "figures" / name
        figures[name] = {"exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0,
                         "sha256": sha256(path) if path.is_file() else None}
    checks["all_four_metric_figures_exist"] = all(item["exists"] and item["bytes"] > 0 for item in figures.values())

    result = {
        "status": "PASS" if files_pass and archive_pass and environment_pass and all(checks.values()) else "FAIL",
        "started": started,
        "finished": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "purpose": "Verify frozen engineering baseline integrity after stage-1 read-only audit",
        "source_files": {"status": "PASS" if files_pass else "FAIL", "manifest_count": len(manifest),
                         "unique_paths": unique_paths, "checked_bytes": bytes_checked,
                         "mismatches": results, "manifest_sha256": sha256(FROZEN / "manifest.json")},
        "snapshot": {"status": "PASS" if archive_pass else "FAIL", "bytes": archive_bytes,
                     "sha256": archive_sha, "method": "Full ZIP-file SHA256 and byte-count comparison to baseline_v0/record.json; archive entries were individually verified at creation"},
        "environment": {"status": "PASS" if environment_pass else "FAIL", "command": ".venv/Scripts/python.exe -m pip freeze --all",
                        "returncode": run.returncode, "frozen_package_count": len(frozen_lines),
                        "current_package_count": len(current_lines),
                        "added_or_changed_entries": sorted(set(current_lines) - set(frozen_lines)),
                        "removed_or_changed_entries": sorted(set(frozen_lines) - set(current_lines)),
                        "stderr": run.stderr.strip(),
                        "method": "Compare sorted nonempty pip-freeze entries; runtime binaries not byte-for-byte attested"},
        "saved_metrics": {"status": "PASS" if all(checks.values()) else "FAIL", "metrics_report_sha256": sha256(metrics_path),
                          "inputs": inputs, "independent_checks": checks, "recomputed_metrics": recomputed,
                          "mean_training_loss": mean_loss, "figures": figures,
                          "scope": "Review stored assertion statuses and independently recompute aggregate arithmetic; no new model inference or training"},
        "limitations": ["Integrity does not establish original-video identity, label semantics, or source-independent validation.",
                        "Top-1 20% is an engineering-split result, not a paper reproduction result or generalization estimate.",
                        "Version-list equality does not prove every environment binary is unchanged; snapshot contains version records, not a complete portable environment."]
    }
    (AUDIT / "final_integrity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "files_checked": len(manifest), "file_mismatches": len(results),
                      "snapshot_sha256": archive_sha, "environment": result["environment"]["status"],
                      "packages": len(current_lines), "metrics": result["saved_metrics"]["status"]}, indent=2), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
