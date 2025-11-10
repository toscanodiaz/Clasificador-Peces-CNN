"""
pip install torch torchvision torchaudio scikit-learn tqdm

"""

import os, time, json, random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# modelo
class CNN(nn.Module):
    """
    [Conv-BN-ReLU]*2 + MaxPool  x5  -> GAP -> FC
    Entrada esperada: 3x224x224
    """
    def __init__(self, num_classes: int, base_ch: int = 32, dropout_p: float = 0.3):
        super().__init__()
        C = base_ch

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            block(3,    C),
            block(C,   C*2), 
            block(C*2, C*4), 
            block(C*4, C*8), 
            block(C*8, C*8), 
        )
        self.dropout = nn.Dropout(dropout_p)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C*8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=1, total_epochs=1):
    model.train()
    total, correct, running = 0, 0, 0.0
    pbar = tqdm(loader, desc=f"Train [{epoch}/{total_epochs}]", leave=True, ncols=100)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and scaler.is_enabled():
            with autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        running += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        pbar.set_postfix({"loss": f"{running/max(1,total):.4f}",
                          "acc": f"{correct/max(1,total):.4f}"})
    return running/total, correct/total


@torch.no_grad()
def evaluate(model, loader, criterion, device, full=False, class_names=None, phase_name="Val", epoch=1, total_epochs=1):
    model.eval()
    total, correct, running = 0, 0, 0.0
    all_p, all_y = [], []
    pbar = tqdm(loader, desc=f"{phase_name} [{epoch}/{total_epochs}]", leave=True, ncols=100)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        running += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        pbar.set_postfix({"loss": f"{running/max(1,total):.4f}",
                          "acc": f"{correct/max(1,total):.4f}"})

        if full:
            all_p.append(preds.cpu().numpy())
            all_y.append(y.cpu().numpy())

    metrics = {"loss": running/total, "acc": correct/total}
    if full and all_p:
        import numpy as np
        p = np.concatenate(all_p); t = np.concatenate(all_y)
        macro = f1_score(t, p, average="macro")
        cm = confusion_matrix(t, p).tolist()
        rep = classification_report(t, p, target_names=class_names, digits=4)
        metrics.update({"macro_f1": float(macro), "confusion_matrix": cm, "classification_report": rep})
    return metrics


def train_and_eval(
    root="fish_dataset",
    subdir="FishImgDataset",
    epochs=25,
    batch_size=32,
    img_size=224,
    lr=3e-4,
    weight_decay=1e-4,
    step_size=8,
    gamma=0.5,
    label_smoothing=0.0,
    patience=7,
    num_workers=0, 
    seed=42,
    output_dir="./outputs",
):

    set_seed(seed)
    device = torch.device("cpu")
    print(f"device {device}")

    # rutas
    data_root = os.path.join(root, subdir)
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")
    test_dir  = os.path.join(data_root, "test")
    for p in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"no existe carpeta {p}")

    # transformaciones
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.15, 0.02),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # datasets / loaders
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(test_dir,  transform=eval_tfms)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f">>> Clases ({num_classes}): {class_names}")

    PIN_MEMORY = False 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=PIN_MEMORY)

    # modelo 
    model = CNN(num_classes=num_classes, base_ch=32, dropout_p=0.3).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    USE_AMP = torch.cuda.is_available()
    scaler = GradScaler("cuda", enabled=USE_AMP)

    # Salidas
    os.makedirs(output_dir, exist_ok=True)
    run = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_path = os.path.join(output_dir, f"{run}_best.pt")
    log_path  = os.path.join(output_dir, f"{run}_log.json")

    best_val, best_epoch, patience_ctr = -1.0, -1, 0
    history = {"train": [], "val": []}
    state = {}

    # loop de entrenamiento
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            epoch=epoch, total_epochs=epochs
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device,
            full=False, class_names=None, phase_name="Val",
            epoch=epoch, total_epochs=epochs
        )

        scheduler.step()

        history["train"].append({"epoch": epoch, "loss": tr_loss, "acc": tr_acc})
        history["val"].append({"epoch": epoch, "loss": val_metrics["loss"], "acc": val_metrics["acc"]})

        print(f"Train | loss={tr_loss:.4f} acc={tr_acc:.4f}")
        print(f"Val   | loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} (time {time.time()-t0:.1f}s)")

        # early stopping 
        if val_metrics["acc"] > best_val:
            best_val, best_epoch, patience_ctr = val_metrics["acc"], epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_val_acc": best_val,
                "class_names": class_names,
                "hparams": {
                    "root": root, "subdir": subdir, "epochs": epochs, "batch_size": batch_size,
                    "img_size": img_size, "lr": lr, "weight_decay": weight_decay,
                    "step_size": step_size, "gamma": gamma, "label_smoothing": label_smoothing,
                    "patience": patience, "num_workers": num_workers, "seed": seed
                }
            }, ckpt_path)
            print(f"mejor modelo guardado en {ckpt_path} (val_acc={best_val:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"early stopping en epoch {epoch} (mejor val_acc={best_val:.4f} @ {best_epoch})")
                break

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"history": history, "best_val_acc": best_val, "best_epoch": best_epoch}, f, indent=2)

    # cargar mejor checkpoint
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        print(f"mejor checkpoint cargado (val_acc={state.get('best_val_acc', 0):.4f})")

    # evaluación completa
    print("\nVALIDACIÓN")
    val_full = evaluate(model, val_loader, criterion, device, full=True, class_names=class_names, phase_name="Val-Full")
    print(f"val  | loss={val_full['loss']:.4f} acc={val_full['acc']:.4f} macroF1={val_full['macro_f1']:.4f}")
    print(val_full["classification_report"])

    print("\nTEST")
    test_full = evaluate(model, test_loader, criterion, device, full=True, class_names=class_names, phase_name="Test-Full")
    print(f"test | loss={test_full['loss']:.4f} acc={test_full['acc']:.4f} macroF1={test_full['macro_f1']:.4f}")
    print(test_full["classification_report"])

    # guardado de métricas
    metrics_path = os.path.join(output_dir, f"{run}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "val": val_full, "test": test_full,
            "best_val_acc": state.get("best_val_acc", best_val) if state else best_val,
            "best_epoch": best_epoch,
            "class_names": class_names,
            "hparams": {
                "root": root, "subdir": subdir, "epochs": epochs, "batch_size": batch_size,
                "img_size": img_size, "lr": lr, "weight_decay": weight_decay,
                "step_size": step_size, "gamma": gamma, "label_smoothing": label_smoothing,
                "patience": patience, "num_workers": num_workers, "seed": seed
            }
        }, f, indent=2)

    print(f"\nmétricas guardadas en {metrics_path}")
    print(f"checkpoint mejor modelo en {ckpt_path}")

    return {
        "checkpoint": ckpt_path,
        "metrics_json": metrics_path,
        "best_val_acc": float(best_val),
        "best_epoch": int(best_epoch),
        "classes": class_names
    }
