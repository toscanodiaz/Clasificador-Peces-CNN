# ------- **EN PROGRESO** -------


Se entrenó un clasificador CNN para reconocer 31 especies de peces del Fish Dataset (Kaggle). Tras tres iteraciones de entrenamiento, la métrica de validación pasó de 52% a 89% y finalmente a 94.7% de exactitud, con macro-F1 0.937 en la corrida final (50 épocas). En esa última corrida, el mejor punto se alcanzó en la época 49 con val loss 0.2391 y val acc 0.9469.

# Problema y datos

Dataset: “Fish Dataset” de Kaggle (imágenes organizadas por especie). Conjunto dividido en train/val/test.

Métricas principales: accuracy en validación y macro-F1. 

# Modelo

Arquitectura 
[Conv-BN-ReLU]×2 + MaxPool (repetido 5 bloques) → GAP → Dropout → FC.

Donde los bloques conv extraen jerarquías de características, GAP convierte mapas de activación en un vector robusto, Dropout regulariza y la capa lineal produce las 31 logits.

Entrenamiento: AdamW, StepLR(step_size=8, gamma=0.5), CrossEntropy.

Transformaciones: resize/crop aleatorio, rotación, flips, jitter de color y normalización tipo ImageNet.

-- gráficas --

# Iteración 1

5 épocas (sin patience, num_workers=0)

Resultados (val): loss 1.6844, acc 0.5162, macro-F1 0.4318.

El modelo aprende rasgos básicos pero está subentrenado con solo 5 épocas y no alcanzan a estabilizar la generalización. Las augmentaciones ayudan un poco pero el número de updates es limitado.

# Iteración 2 

25 épocas (patience=7, num_workers=4)

Resultados (val): loss 0.4195, acc 0.8902, macro-F1 0.8750.

Al haber subido las épocas el modelo tuvo tiempo de aprender features de mayor nivel como bordes, texturas, patrones etc.

DataLoader más rápido (num_workers=4) permite batches más constantes y entrenamiento más estable.

StepLR con 25 épocas ya hubo tres decaimientos del LR (8, 16, 24) lo que tiende a mejorar la convergencia.

# Iteración 3 

50 épocas (patience=7, num_workers=4/0 en la última corrida)

Resultados (val): loss 0.2391, acc 0.9469, macro-F1 0.9370, mejor época: 49.
