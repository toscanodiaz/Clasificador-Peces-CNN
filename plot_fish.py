""" 
python plot_fish.py --log "./outputs/output_20251108_145328_log.json" --metrics "./outputs/output_20251108_145328_metrics.json"

"""

import json, os, argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="ruta a *_log.json")
    ap.add_argument("--metrics", required=True, help="ruta a *_metrics.json")
    ap.add_argument("--outdir", default=None, help="carpeta de salida")
    args = ap.parse_args()

    with open(args.log, "r", encoding="utf-8") as f:
        log = json.load(f)
    with open(args.metrics, "r", encoding="utf-8") as f:
        mets = json.load(f)

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.log))
    os.makedirs(outdir, exist_ok=True)

    hist_df = pd.DataFrame({
        "epoch": [e["epoch"] for e in log["history"]["train"]],
        "train_loss": [e["loss"] for e in log["history"]["train"]],
        "train_acc":  [e["acc"]  for e in log["history"]["train"]],
        "val_loss":   [e["loss"] for e in log["history"]["val"]],
        "val_acc":    [e["acc"]  for e in log["history"]["val"]],
    })
    hist_csv = os.path.join(outdir, "history.csv")
    hist_df.to_csv(hist_csv, index=False)

    # gráficas
    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_acc"], label="Train ACC")
    plt.plot(hist_df["epoch"], hist_df["val_acc"],   label="Val ACC")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("accuracy por época")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "acc.png"), dpi=160, bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
    plt.plot(hist_df["epoch"], hist_df["val_loss"],   label="Val Loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("loss por época")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "loss.png"), dpi=160, bbox_inches="tight"); plt.close()

    # matrices de confusión 
    class_names = mets["val"].get("classification_report","").splitlines()[0:0]
    cm_val = np.array(mets["val"]["confusion_matrix"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_val)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("matriz de confusión - Val")
    plt.savefig(os.path.join(outdir, "cm_val.png"), dpi=160, bbox_inches="tight"); plt.close()

    cm_test = np.array(mets["test"]["confusion_matrix"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap="Greens", colorbar=False)
    plt.title("matriz de confusión - test")
    plt.savefig(os.path.join(outdir, "cm_test.png"), dpi=160, bbox_inches="tight"); plt.close()

    print(f"guardado en {outdir}")

if __name__ == "__main__":
    main()
