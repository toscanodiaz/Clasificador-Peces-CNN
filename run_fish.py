"""
python run_fish.py --root "./fish_dataset" --subdir "FishImgDataset" --epochs 25 --batch-size 32 --img-size 224 --num-workers 0

python run_fish.py --root "./fish_dataset" --subdir "FishImgDataset" --epochs 50 --patience 5 --batch-size 32 --img-size 224 --num-workers 4

"""


import argparse
from model_fish import train_and_eval

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="fish_dataset", help="ruta base que contiene FishImgDataset/")
    p.add_argument("--subdir", type=str, default="FishImgDataset", help="carpeta del dataset dentro de root")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--step-size", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="./outputs")
    args = p.parse_args()

    res = train_and_eval(
        root=args.root, subdir=args.subdir,
        epochs=args.epochs, batch_size=args.batch_size, img_size=args.img_size,
        lr=args.lr, weight_decay=args.weight_decay,
        step_size=args.step_size, gamma=args.gamma,
        label_smoothing=args.label_smoothing,
        patience=args.patience, num_workers=args.num_workers,
        seed=args.seed, output_dir=args.output_dir
    )

    print("\nRESUMEN")
    print(f"mejor ACC VAL: {res['best_val_acc']:.4f} @ epoch {res['best_epoch']}")
    print(f"mheckpoint: {res['checkpoint']}")
    print(f"métricas: {res['metrics_json']}")

if __name__ == "__main__":
    main()
