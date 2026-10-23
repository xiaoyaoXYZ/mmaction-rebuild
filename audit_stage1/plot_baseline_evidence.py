"""Recompute and plot saved baseline_v0 evidence; never run model training.

Run with the project's existing .venv Python. Inputs are read-only; all outputs
are written beside this script. This is an arithmetic consistency check, not an
independent validation experiment.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"
INPUTS = {
    "train": ROOT / "work_dirs/train_gpu_01/result.json",
    "evaluate": ROOT / "work_dirs/evaluate_gpu_01/result.json",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    hashes_before = {key: digest(path) for key, path in INPUTS.items()}
    train, evaluate = [
        json.loads(INPUTS[key].read_text(encoding="utf-8"))
        for key in ("train", "evaluate")
    ]
    checks = []

    def check(condition, description):
        if not bool(condition):
            raise AssertionError(description)
        checks.append({"assertion": description, "status": "PASS"})

    check(train["status"] == evaluate["status"] == "PASS", "Both saved runs report PASS")
    iterations = np.array([item["iteration"] for item in train["iterations"]])
    losses = np.array([item["metrics"]["loss"] for item in train["iterations"]])
    check(np.array_equal(iterations, np.arange(1, 11)), "Exactly ten recorded iterations, numbered 1 through 10")
    check(np.isfinite(losses).all(), "All ten recorded training losses are finite")
    check(train["epochs"] == 1, "Saved run contains one training epoch")
    check(
        all(item["metrics"]["loss"] == item["metrics"]["loss_cls"] for item in train["iterations"]),
        "Each saved total loss equals the saved classification loss",
    )
    check(train["config_sha256"] == evaluate["config_sha256"], "Training and evaluation record the same configuration hash")
    check(train["checkpoint"]["sha256"] == evaluate["checkpoint"]["sha256"], "Training and evaluation record the same checkpoint hash")

    clips = evaluate["clips"]
    classes = [evaluate["classes"][str(i)] for i in range(10)]
    probabilities = np.array([clip["probabilities"] for clip in clips], dtype=np.float64)
    labels = np.array([clip["true_label"] for clip in clips], dtype=np.int64)
    check(probabilities.shape == (10, 10), "Saved evaluation contains ten clips and ten probabilities per clip")
    check(np.isfinite(probabilities).all(), "All saved probabilities are finite")
    check(((probabilities >= 0) & (probabilities <= 1)).all(), "All saved probabilities are within [0, 1]")
    check(np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6), "Probability row sums equal one within absolute tolerance 1e-6")
    check(np.array_equal(np.sort(labels), np.arange(10)), "Engineering validation contains exactly one sample for every class")
    check(len({clip["clip"] for clip in clips}) == 10, "Saved evaluation contains ten distinct clip path strings")
    check(
        all(clip["true_class"] == classes[clip["true_label"]] for clip in clips),
        "Saved true class strings agree with the saved integer-to-class map",
    )
    val_rows = {item["clip"]: item for item in evaluate["dataset"]["val"]["rows"]}
    check(
        set(val_rows) == {clip["clip"] for clip in clips}
        and all(val_rows[clip["clip"]]["label"] == clip["true_label"]
                and val_rows[clip["clip"]]["frames"] == clip["total_frames"] for clip in clips),
        "Saved clip labels and frame counts agree with the evaluation annotation summary",
    )
    check(train["dataset"]["val"] == evaluate["dataset"]["val"], "Training and reloaded evaluation use the same saved validation summary")
    predictions = probabilities.argmax(axis=1)
    rankings = np.argsort(-probabilities, axis=1, kind="stable")
    check(np.all(np.diff(np.sort(probabilities, axis=1), axis=1) > 0), "No equal probability values within a clip; top-k tie policy does not affect this result")
    check(np.array_equal(predictions, [clip["predicted_label"] for clip in clips]), "Probability argmax agrees with every saved predicted label")
    check(
        all(clip["predicted_class"] == classes[clip["predicted_label"]] for clip in clips),
        "Saved predicted class strings agree with the saved integer-to-class map",
    )
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    support = confusion.sum(axis=1)
    per_class_accuracy = confusion.diagonal() / support
    top1 = float((predictions == labels).mean())
    top5 = float((rankings[:, :5] == labels[:, None]).any(axis=1).mean())
    mean_class = float(per_class_accuracy.mean())
    recomputed = {"acc/top1": top1, "acc/top5": top5, "acc/mean1": mean_class}
    check(int(confusion.sum()) == 10, "Count confusion matrix totals ten samples")
    check(np.array_equal(confusion, evaluate["confusion_matrix"]), "Recomputed count confusion matrix equals the saved matrix")
    for name, value in recomputed.items():
        for input_key, document in (("train", train), ("evaluate", evaluate)):
            check(abs(value - document["validation_metrics"][name]) <= 1e-12,
                  f"Recomputed {name} agrees with {input_key} result within absolute tolerance 1e-12")

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(9.2, 5.0), layout="constrained")
    ax.plot(iterations, losses, color="#19628c", marker="o", linewidth=1.7)
    ax.set(xticks=iterations, xlabel="Recorded optimizer iteration (one epoch)", ylabel="Training classification loss",
           title="baseline_v0: 10 recorded training losses\nEngineering split / NOT independent validation")
    ax.grid(axis="y", alpha=0.25)
    fig.supxlabel("Raw logged values; no smoothing, no validation-loss series, no convergence claim.", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"baseline_v0_loss.{suffix}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 9.0), layout="constrained")
    im = ax.imshow(confusion, cmap="Blues", vmin=0, vmax=1)
    ticks = [f"{i}: {name}" for i, name in enumerate(classes)]
    ax.set(xticks=np.arange(10), yticks=np.arange(10), xticklabels=ticks, yticklabels=ticks,
           xlabel="Predicted class", ylabel="True class",
           title="baseline_v0: confusion matrix (counts; one sample per true class)\nEngineering split / NOT independent validation")
    plt.setp(ax.get_xticklabels(), rotation=48, ha="right", rotation_mode="anchor")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", color="white" if confusion[i, j] else "#405060", fontsize=9)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 1])
    colorbar.set_label("Sample count")
    fig.supxlabel("10 engineering-validation clips; source independence unconfirmed. No paper reproduction claim.", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(FIGURES / f"baseline_v0_confusion.{suffix}", dpi=180)
    plt.close(fig)

    check(hashes_before == {key: digest(path) for key, path in INPUTS.items()}, "Both input result JSON files remain byte-for-byte unchanged")
    report = {
        "status": "PASS",
        "created": datetime.now().astimezone().isoformat(),
        "purpose": "Arithmetic consistency check of saved engineering baseline evidence; not an independent validation experiment",
        "runtime": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "inputs": {key: {"path": str(path.relative_to(ROOT)), "sha256": hashes_before[key]} for key, path in INPUTS.items()},
        "class_map_as_recorded": dict(enumerate(classes)),
        "recorded_training": {"epochs": 1, "iterations": iterations.tolist(), "loss": losses.tolist(), "mean_loss": float(losses.mean()), "validation_loss_available": False},
        "recomputed_engineering_metrics": recomputed,
        "count_confusion_matrix": confusion.tolist(),
        "confusion_orientation": "rows=true labels, columns=predicted labels",
        "class_support": support.tolist(),
        "per_class_accuracy": per_class_accuracy.tolist(),
        "prediction_counts": np.bincount(predictions, minlength=10).tolist(),
        "samples": [{"clip": clip["clip"], "true_label": int(labels[i]), "argmax_label": int(predictions[i]), "top5_labels": rankings[i, :5].tolist()} for i, clip in enumerate(clips)],
        "assertions": checks,
        "limitations": [
            "No model execution or new training performed; saved probabilities and losses are the evidence inputs.",
            "Source sessions, persons and original-video independence are unconfirmed. This script does not establish a trustworthy split.",
            "Exactly one engineering-validation sample per class; metrics do not establish generalization or paper reproduction.",
            "Ten loss points from one epoch do not establish convergence, overfitting success or a multi-epoch trend.",
            "Class strings are reproduced as recorded, not adjudicated semantic ground truth; label 0 is named anya_mopping while clip paths use an_mopping.",
        ],
    }
    (OUT / "metrics_recheck.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    notes = f"""# baseline_v0 既有数值证据复核

本次只读取既有 train/evaluate 的 result.json，使用现有 .venv 中 NumPy {np.__version__} 与 Matplotlib {matplotlib.__version__} 重算指标并绘图。没有运行模型、训练、安装依赖或修改旧结果。这里的“重算”是计算过程独立的算术核对，**不代表独立验证集评估**。

## 结果与成功判据

- {len(checks)} 项断言全部通过；逐条记录在 `metrics_recheck.json` 的 `assertions`。
- 10 个实际训练迭代，loss 全部有限，首项 {losses[0]:.6f}、末项 {losses[-1]:.6f}，均值 {losses.mean():.6f}。图保留原始波动，无平滑；未生成不存在的 validation loss 或多轮趋势。
- 从 10 个已保存概率向量与 true label 重算：Top-1 = {top1:.1%}（2/10）、Top-5 = {top5:.1%}（8/10）、mean class accuracy = {mean_class:.1%}。与训练时验证及重新加载模型后的既有记录均一致。
- 混淆矩阵按“行=真实类、列=预测类”重建并逐元素核对一致。每类只有 1 个工程验证样本；预测类别 0 和 7 各 5 个，其余类别均为 0 个预测。该现象只描述这次输出，不能据此推断模型已经收敛或判定长期模型行为。
- 输入 SHA-256 在执行前后相同；保留在 JSON。训练和评估记录中的配置/权重哈希相同。本脚本未重新读取权重或复核权重文件本身，权重完整性属于 baseline_v0 快照校验范围。

## 文件

- `figures/baseline_v0_loss.png` / `.svg`：10 次原始训练 loss。
- `figures/baseline_v0_confusion.png` / `.svg`：10×10 计数混淆矩阵，不做行归一化；两图均明确标记 Engineering split / NOT independent validation。
- `metrics_recheck.json`：原始 loss、重新计算的指标/混淆矩阵、每类支持数、每样本 top-5、数据源哈希与所有断言。

## 局限与后续前提

- 原始视频、采集会话、人员及训练/验证之间来源独立性尚未由本脚本确认；请以本轮 `data_audit.md` 的来源与重复帧审计为准。现有 20% Top-1 仅是工程划分上的流程证据，**不是泛化性能，也不是论文复现结果**。
- 每类仅 1 个验证样本，不能稳定估计每类表现。只有 10 个训练 loss 点且末值高于首值，不能宣称收敛或完成少样本过拟合测试。
- 当前类别字符串原样取自保存结果。编号 0 的 `anya_mopping` 与路径 `an_mopping` 命名不同，本脚本不作语义裁决。
- 少量训练样本过拟合测试尚未执行；应另设输出目录，固定训练样本与确定性采样，记录 loss、训练样本命中情况与梯度/参数更新；不能把同一批样本称为独立验证。
- 需要来源清单、人工标签确认以及按原始素材分组的可信划分后，才能新增独立验证、loss 曲线和混淆矩阵实验。样本不足时应先补数据。

## 重跑

在 `mmaction_rebuild` 目录执行：

```powershell
& '.\\.venv\\Scripts\\python.exe' '.\\audit_stage1\\plot_baseline_evidence.py'
```

脚本只覆盖自身生成的审计图、metrics_recheck.json 与本说明，不改动输入结果、代码、配置、环境、原始数据或模型权重。
"""
    (OUT / "evidence_notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps({"status": "PASS", "assertions": len(checks), "engineering_metrics": recomputed, "outputs": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
