Momento de Retroalimentación: Módulo 2 Implementación de un modelo de deep learning. (Portafolio Implementación)

Ana Karen Toscano Díaz A01369687

<img width="567" height="193" alt="image" src="https://github.com/user-attachments/assets/a522414a-cd13-4386-89ad-aebdb66b56f8" />

# Clasificador Peces CNN

Este proyecto implementa una red neuronal convolucional para clasificar imágenes de diferentes especies de peces. Se utiliza PyTorch y Torchvision para el procesamiento, entrenamiento y evaluación e implementa la visualización de métricas de aprendizaje en tiempo real.

Video demostración del clasificador: [fishclassifier.mp4](https://drive.google.com/file/d/1iD0jYUkxk3ZaiCycANYzXOzhqoa_52F3/view?usp=sharing)


## Estructura del proyecto

**evidenciaDL**/
- run_fish.py --> script de entrenamiento
- model_fish.py --> creación, entrenamiento y evaluación del modelo
- plot_fish.py --> graficar accuracy/loss y métricas finales

   **outputs**/ --> carpeta donde se guardan resultados de mejor modelo
   - output_0251108_145328_best.pt
   - output_0251108_145328_log.json
   - output_0251108_145328_metrics.json
   
   **fish_dataset**/
   - FishImgDataset/
      - train/
      - val/
      - test/

**Reporte.md** --> documentación del proyecto

## Descripción general

Se utilizó el dataset de Kaggle [Fish Dataset](https://www.kaggle.com/datasets/markdaniellampa/fish-dataset). 

Flujo completo

1. Entrenamiento con augmentaciones (rotación, color, flips, etc) 
2. Validación por época con early stopping
3. Métricas finales de evaluación 
4. Resultados guardados
   - Checkpoint del mejor modelo (.pt)
   - Métricas finales (_metrics.json)
   - Historial por época (_log.json)
5. Visualización del aprendizaje y desempeño (plot_fish.py)


## Arquitectura del modelo

[Conv-BN-ReLU] *2 + MaxPool ×5 → GAP → Dropout → Fully Connected

- Función de pérdida: CrossEntropyLoss  
- Optimizador: AdamW  
- Scheduler: StepLR con decaimiento exponencial del learning rate  


## Requisitos

Instalar dependencias:

```bash
pip install torch torchvision torchaudio scikit-learn tqdm matplotlib
```


## Entrenamiento

```bash
python run_fish.py --root "./fish_dataset" --subdir "FishImgDataset" \
                   --epochs 50 --patience 7 \
                   --batch-size 32 --img-size 224 --num-workers 4
```

Durante el entrenamiento se muestran métricas en tiempo real con tqdm


## Visualización de resultados

Después del entrenamiento se pueden graficar las curvas de aprendizaje y las métricas

```bash
python plot_fish.py --log "./outputs/output_20251108_145328_log.json" \
                    --metrics "./outputs/output_20251108_145328_metrics.json"
```

Correr la interfaz gráfica:

```bash
python UI_fish.py --ckpt "./outputs/output_20251108_145328_best.pt"
```

## Archivos generados

- *_best.pt	--> pesos del mejor modelo (checkpoint)

- *_metrics.json	--> resultados finales en val/test, F1, matriz de confusión

- *_log.json	--> histórico de entrenamiento por época
