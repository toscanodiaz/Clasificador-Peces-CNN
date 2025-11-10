# Clasificador-Peces-CNN
Momento de Retroalimentación: Módulo 2 Implementación de un modelo de deep learning. (Portafolio Implementación)

# Fish Classifier CNN – Deep Learning con PyTorch

Este proyecto implementa una red neuronal convolucional para clasificar imágenes de diferentes especies de peces. Se utiliza PyTorch y Torchvision para el procesamiento, entrenamiento y evaluación, también early stopping para guardar el mejor modelo y la visualización de métricas de aprendizaje.

## Descripción general

El modelo aprende a reconocer especies de peces a partir del dataset de Kaggle [Fish Dataset](https://www.kaggle.com/datasets/markdaniellampa/fish-dataset). 

El flujo completo del proyecto es el siguiente:

1. Entrenamiento con augmentaciones de datos (rotación, color, flips, etc) 
2. Validación por época con early stopping
3. Métricas finales de evaluación 
4. Resultados guardados:
   - Checkpoint del mejor modelo (.pt)
   - Métricas finales (_metrics.json)
   - Historial por época (_log.json)
5. Visualización del aprendizaje y desempeño (plot_fish.py)


## Arquitectura del modelo

La CNN tiene la siguiente estructura:

[Conv-BN-ReLU] *2 + MaxPool ×5 → GAP → Dropout → Fully Connected

- Entrada esperada: 3×224×224  
- Salidas: número de clases (31 especies en el dataset)  
- Función de pérdida: CrossEntropyLoss  
- Optimizador: AdamW  
- Scheduler: StepLR con decaimiento exponencial del learning rate  


## Requisitos

Instalar dependencias:

```bash
pip install torch torchvision torchaudio scikit-learn tqdm matplotlib
```


## Estructura del proyecto

evidenciaDL/
│
├── run_fish.py             # sript de entrenamiento
├── model_fish.py           # creación, entrenamiento y evaluación del modelo
├── plot_fish.py            # gráficar accuracy/loss y métricas finales
│
├── outputs/                # carpeta donde se guardan resultados de mejor modelo
│   ├── output_0251108_145328_best.pt
│   ├── output_0251108_145328_log.json
│   └── output_0251108_145328_metrics.json
│
└── fish_dataset/
    └── FishImgDataset/
        ├── train/
        ├── val/
        └── test/


## Entrenamiento

```bash
python run_fish.py --root "./fish_dataset" --subdir "FishImgDataset" \
                   --epochs 50 --patience 7 \
                   --batch-size 32 --img-size 224 --num-workers 4
```

Durante el entrenamiento se muestran métricas en tiempo real con tqdm.


## Visualización de resultados

Después del entrenamiento, se pueden graficar las curvas de aprendizaje y las métricas:

```bash
python plot_fish.py --log "./outputs/output_20251108_145328_log.json" \
                    --metrics "./outputs/output_20251108_145328_metrics.json"
```

## Archivos generados

- *_best.pt	--> Pesos del mejor modelo (checkpoint)

- *_metrics.json	--> Resultados finales: val/test, F1, matriz de confusión

- *_log.json	--> Historial completo de entrenamiento por época
