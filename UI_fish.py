"""
python UI_fish.py --ckpt "./outputs/output_20251108_145328_best.pt"

pip install gradio pillow

"""

import argparse, torch, torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr
import os, json

class SimpleDeepCNN(nn.Module):
    def __init__(self, num_classes: int, base_ch: int = 32, dropout_p: float = 0.3):
        super().__init__()
        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256), block(256, 256)
        )
        self.dropout = nn.Dropout(0.3)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)

def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    class_names = state["class_names"]
    model = SimpleDeepCNN(num_classes=len(class_names))
    model.load_state_dict(state["model_state"])
    model.eval().to(device)
    return model, class_names

IM_SIZE = 224
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
eval_tfms = transforms.Compose([
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="ruta al checkpoint .pt")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.ckpt, device)

    def predict(img: Image.Image):
        if img.mode != "RGB": img = img.convert("RGB")
        x = eval_tfms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return {cls: float(probs[i]) for i, cls in enumerate(class_names)}

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="sube una imagen"),
        outputs=gr.Label(num_top_classes=3, label="predicción (top 3)"),
        title="CLASIFICADOR DE PECES",
        # description="clasificador de peces",
        theme=gr.themes.Soft(
            primary_hue="pink",
            secondary_hue="orange",
        )
    )
    demo.launch(share=args.share, server_port=args.server_port)

if __name__ == "__main__":
    main()
