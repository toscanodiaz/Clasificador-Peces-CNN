# ------- **EN PROGRESO** -------

<img width="483" height="237" alt="image" src="https://github.com/user-attachments/assets/3a4e6f15-8838-408b-bf5d-6378a23be749" />

---

# Problema y datos

Se entrenó un clasificador multiclase para reconocer 31 especies de peces del Fish Dataset (Kaggle). Tras tres iteraciones de entrenamiento, la métrica de validación pasó de 52% a 89% y finalmente a 94.7% de exactitud, con macro-F1 0.937 en la corrida final (50 épocas). En esa última corrida, el mejor punto se alcanzó en la época 49 con val loss 0.2391 y val acc 0.9469.

Dataset: “Fish Dataset” de Kaggle (imágenes organizadas por especie). Conjunto dividido en train/val/test. Las imágenes se redimensionaron a 224×224 RGB. 

Métricas principales: accuracy en validación y macro-F1. 

# Modelo

Arquitectura 
[Conv --> BatchNorm --> ReLU] ×2 + MaxPool (repetido 5 bloques) → GAP → Dropout → FC.

Donde los bloques conv extraen jerarquías de características, Global Average Pooling (GAP) convierte mapas de activación en un vector robusto, Dropout (p=0.3) regulariza y la capa totalmente conectada (FC) produce las 31 neuronas de salida (una por clase).

5 bloques tipo [Conv2D → BatchNorm → ReLU] ×2 + MaxPool.

Entrenamiento: AdamW, StepLR(step_size=8, gamma=0.5), CrossEntropy.

Transformaciones: resize/crop aleatorio, rotación, flips, jitter de color y normalización tipo ImageNet.

---

# Iteración 1

## Entrenamiento

- **epochs**: 5
- **batch_size** = 32
- **num_workers** = 0 (carga de datos secuencial) 
- **Optimizador**: AdamW con lr = 3e-4 y weight_decay = 1e-4 
- **Scheduler StepLR**: step_size = 8, gamma = 0.5 (en esta iteración no llega a activarse porque solo hay 5 épocas)
- **Función de pérdida**: CrossEntropyLoss (label_smoothing = 0.0).

## Resultados en val
**loss**: 1.6844, **acc**: 0.5162, **F1**: 0.4318

## Gráficas

### Accuracy por época

<img width="452" height="361" alt="acc1" src="https://github.com/user-attachments/assets/51a4b6d7-ac96-4339-b47e-1f15071cf36f" />

En la gráfica se ve que el accuracy de train y de val suben de manera casi lineal desde la primera época hasta la última, con la validación empezando un poco arriba de train (0.31 vs 0.24) y termina también ligeramente arriba (0.516 vs 0.484 aprox), la separación entre las dos curvas se ve pequeeña y estable. 

Se puede interpretar que el modelo está aprendiendo de forma consistente pues cada época aporta información útil y no hay saltos bruscos ni señales de sobreajuste porque la curva de validación no se dobla hacia abajo en ningún momento. Que la validación llegue a 0.516 indica que la red está capturando patrones relevantes pero el crecimiento lineal demuestra que 5 épocas no alcanzan para estabilizar la generalización --> el modelo está subentrenado y faltan más épocas para que suba la curva. 

### Loss por época

<img width="452" height="361" alt="loss1" src="https://github.com/user-attachments/assets/ee853bc7-5656-4fe3-be58-255facd72f36" />

En esta gráfica se oberva que el loss de train baja de 2.78 a 1.82 aprox y el loss de val baja de 2.48 a 1.68 aprox; las dos curvas tienen una forma muy similar y no se cruzan de manera inusual, solo convergen hacia abajo. 

La disminución suave de loss confirma que la optimización con AdamW y el lr está bien configurada. El loss de val es un poco menor que el loss de train, lo cual pasó porque hay regularización y el modelo aún no ha tenido tiempo de sobreajustar. Además la tendencia hacia abajo indica de nuevo underfitting por lo que se deben de aumentar las épocas en una nueva iteración. 

### Matriz de confusión Validation 

<img width="423" height="434" alt="cm_val1" src="https://github.com/user-attachments/assets/e635d5f4-e384-45f4-aa1f-b72e712f4dd0" />

En la matriz de confusión se ve una diagonal muy visible pero no muy limpia porque hay celdas fuera de la diagonal con valores significativos, algunas clases muestran bloques claros en la diagonal lo cual se refiere a muchos aciertos, pero otras clases tienen filas más dispersas con varios errores repartidos entre las columnas. 

Se puede inferir que el modelo ya es capaz de identificar correctamente ciertas especies con patrones muy distintivos y con suficientes ejemplos, sin embargo la dispersión en las otras filas indica confusiones sentre especies visualmente parecidas como por ejemplo distintos tipos de peces alargados o carpas... como esta matriz corresponde a solo 5 épocas es normal que el modelo aún no haya refinado los filtros por completo, hacen falta más iteraciones. 

### Matriz de confusión Test

<img width="423" height="434" alt="cm_test1" src="https://github.com/user-attachments/assets/07cb3f5e-cfd3-4780-8b59-1a7654d723d3" />

La estructura de esta matriz es muy parecida a la de validación con la diagonal visible pero con ruido, además de que las mismas clases tienden a verse bien o mal. No se observa que empeore considerablemente al pasar de validación a test, en general se mantienen los patrones de acierto y error; esto indica que el modelo generaliza de forma consistente trasladando lo que aprendió en train/val hacia datos que no había visto. El hecho de que tanto en val como en test hayan clases muy fuertes y otras clases débiles demuestra que el problema no es solo overfitting sino también clases desbalanceadas que necesitan más entrenamiento y tal vez más datos. 

---

# Iteración 2 

## Entrenamiento

- **epochs**: 25
- **patience**: 7 (early stopping)
- **batch_size** = 32
- **num_workers** = 4 (data loader más rápido) 
- **Optimizador**: AdamW con lr = 3e-4 y weight_decay = 1e-4 
- **Scheduler StepLR**: step_size = 8, gamma = 0.5
- **Función de pérdida**: CrossEntropyLoss (label_smoothing = 0.0).

## Resultados en val
**loss**: 0.4195, **acc**: 0.8902, **F1**: 0.8750

**Mejor época**: epoch 25 (el entrenamiento llegó hasta el final y no se hizo early stopping).

## Comparativa
Durante la primera iteración, el modelo sólo entrenó 5 épocas y terminó con acc: 0.5162 y F1: 0.4318, indicando un modelo subentrenado pues concluyó el entrenamiento antes de que la red comenzara a aprender patrones. En esta iteración se aumentaron las épocas de 5 a 25 dándole al modelo más tiempo para ajustar todos los pesos de la red convolucional, también se agregó early stopping con patience: 7 para evitar overfitting pero no se activó en esta iteración (nunca se detuvo antes). Igual se aumetó num_workers a 4 para agilizar la lectura de imágenes sin afectar la calidad del modelo. 

En validación el accuracy subió de 0.3166 al inicio hasta 0.8902, lo que valida que la arquitectura definida tiene la suficiente capacidad pero necesitaba más pasos de optimización, el F1: 0.875 indica que el modelo no sólo acierta globlamente sino que tiene un buen rendimiento en casi todas las clases, aún en las menos frecuentes; y el loss bajó de 2.413 a 0.4195 lo que significa que el modelo tiene más aciertos y la confianza es mayor en la clase correcta.
Lo que explica esta mejora de desempeño del modelo es que ahora se entrenó con más épocas y más tiempo de manera estable. 

## Gráficas

### Accuracy por época

<img width="452" height="361" alt="acc2" src="https://github.com/user-attachments/assets/1b9cf04c-6dcf-46b3-8d09-c83254515237" />

La curva de train sube casi monótonamente desde 0.24 hasta 0.87 y la de val también sube consistentemente con pequeños saltos y bajadas pero su tendencia es claramaente hacia arriba, resultando en el mejor modelo de la época 25 con 0.8902 de accuracy. 

En varias épocas la curva de val está ligeramente por encima de la de train lo que sucede por las augmentaciones que hay en train, la dificultad de los ejemplos entre ambas está repartida de forma un poco aleatoria. A partir de la época 16 - 17 hay un salto grande en accuracy de val que sube de 0.7524 a 0.8426 y se estabiliza entre 0.85 - 0.86 hasta llegar a 0.89 pues el modelo termina de ajustar pesos y separa mejor las clases difíciles. 

### Loss por época 

<img width="452" height="361" alt="loss2" src="https://github.com/user-attachments/assets/af294d1c-8010-447d-8434-b2df19022b5f" />

Las dos curvas de train y val disminuyen de forma pronunciada de aprox 2.7 - 2.5 en la época 1 a menos de 0.5 en la época 25, lo que indica que el modelo sigue aprendiendo durante casi todas las épocas. De la primera a la octava época ambas curvas son muy cercanas y casi son paralelas, indicando que el modelo aprende patrones reales y no está sólo memorizando el set de train. 

Entre las épocas 7 - 18 aprox se ven pequeños picos en la curva de val, suben un poco en 13, 15 y 17 pero después vuelve a bajar, esto pasa porque el optimizador se mueve en la superficie de pérdida probando diferentes conjuntos de parámetros para ver cuáles reducen el error y puede tomar un paso que lo lleva a una posición ligeramente peor, o sea un punto un poco más alto momentáneamente, pero a medida que el entrenamiento continúa se vuelve a estabilizar y a caer en un mejor punto más bajo. 

A partir de la época 17 aprox, el loss se mantiene en 0.5 - 0.6 y termina en 0.41 aproximadamente, que coincide con el mejor modelo (epoch 25). La separación de las curvas es poca lo que indica poco overfitting y que el modelo generaliza bien.

### Matriz de confusión Validation

<img width="423" height="434" alt="cm_val2" src="https://github.com/user-attachments/assets/28024247-2450-4b1d-b8bb-a6bd524fec30" />

La diagonal de la matriz tiene muchos valores altos, o sea que el modelo acierta la clase correcta en la mayoría de los casos. Varias clases muestran un comportamiento casi perfecto (Grass Carp, Glass Perchlet, Gourami, Gold Fish, Silver Carp...) con números altos y pocos errores dispersos, sin embargo las confusiones que aparecen normalmanete se dan en las especies con morfología o colores parecidos, lo cual es esperable por la naturaleza del dataset. 

No hay ninguna clase que esté totalmente sin aciertos, confirmando que el modelo aprende todas las clases aunque algunas sean más difíciles que otras, alineado con el macro-F1 alto (0.875) --> el modelo no solo tiene buena accuracy global, sino que también se comporta bien por clase. 

### Matriz de confusión Test

<img width="423" height="434" alt="cm_test2" src="https://github.com/user-attachments/assets/39bb3ea2-34ce-4d02-9485-52401079e4f9" />

El patrón de esta matriz en test es muy similar al de val con diagonales fuertes y errores concentrados en pocas clases, el accuracy en test de aprox 0.88 está muy cerca de la de val: 0.8902 --> el modelo no se sobreajusta al set de val sino que generaliza de verdad. 

Las clases fuertes (Glass Perchlet, Grass Carp, Gourami, Tilapia...) mantienen muchos aciertos, y clases más difíciles (Mudfish, Climbing Perch, Tenpounder...) presentan algunas confusiones pero aun así con un número decente de aciertos.

El comportamiento en test confirma que el modelo de esta segunda iteración ya es un clasificador útil y estable en lugar de sólo un experimento en train/val.

---

# Iteración 3 

50 épocas (patience=7, num_workers=4/0 en la última corrida)

Resultados (val): loss 0.2391, acc 0.9469, macro-F1 0.9370, mejor época: 49.
