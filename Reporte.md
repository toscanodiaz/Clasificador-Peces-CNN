# FISH CLASSIFIER 

<img src="https://github.com/user-attachments/assets/8451c155-1ba3-4cf3-ba50-0fc3c4ddc82a" alt="imagen" width="552" />



---

# Problema y datos

Se entrenó un clasificador multiclase para reconocer 31 especies de peces del Fish Dataset de Kaggle. Después de tres iteraciones de entrenamiento, el mejor punto se alcanzó en la época 49 de la tercera iteración (50 épocas), con 94.7% de exactitud, val loss de 0.2391 y F1 de 0.937. 

**Dataset**: [Fish Dataset](https://www.kaggle.com/datasets/markdaniellampa/fish-dataset) de Kaggle (imágenes organizadas por especie). Conjunto dividido en train/val/test. Las imágenes se redimensionaron a 224×224 RGB. 

**Métricas principales**: accuracy en validación y F1. 

# Modelo

## Arquitectura

[Conv --> BatchNorm --> ReLU] ×2 + MaxPool (repetido 5 bloques) 

--> Global Average Pooling (GAP) 

--> Dropout (p = 0.3) 

--> Fully Connected (FC)

### Bloques convolucionales
Cada bloque tiene dos capas convolucionales consecutivas con kernel 3×3 seguidas de Batch Normalization y activación ReLU:

- **Conv2D 3×3**: extraen patrones visuales de manera jerárquica
  - El primer bloque corresponde a bordes y texturas simples
  - Los bloques intermedios a patrones más abstractos
  - Los bloques finales son las formas completas y estructuras propias de las especies de peces
- **BatchNorm**: normaliza activaciones y estabiliza el entremamiento permitiendo aprender más rápido y reduciendo sensibilidad a pesos iniciales.
- **ReLU**: introduce no linealidad para que el modelo aprenda funciones complejas.
- **MaxPool 2×2**: hace downsampling (reduce la resolución espacial progresivamente) y permite que las capas profundas vean el contexto global de la imagen. 

Estos bloques producen mapas de características profundos con la información visual necesaria para distinguir a las especies con diferencias pequeñas entre sí (formas de aletas, colores, patrones en el cuerpo).

### Global Average Pooling (GAP)
En lugar de utilizar capas densas intermedias grandes que aumentan el riesgo de sobreajuste, GAP reduce cada mapa de características a un solo número, o sea su promedio, lo que produce un vector compacto que resume la activación global de cada una de las features. Reduce drásticamente los parámetros, favorece la generalización y evita overfitting. 

### Dropout (p = 0.3)
Antes de la capa final se aplica Dropout para desactivar aleatoriamente el 30% de las neuronas en cada paso de entrenamiento, lo que obliga a la red a no depender de neuronas específicas y mejora la capacidad de generalización. 

### Fully Connected (capa final)
Capa densa con 31 neuronas (una por especie de pez). Se combina con el CrossEntropyLoss que internamente aplica un softmax para producir probabilidades por clase.

## Entrenamiento

### Optimizador AdamW
Se eligió AdamW porque además de ser el optimizador estándar para CNNs modernas maneja bien la regularización mediante weight decay desacoplado, suaviza oscilaciones del gradiente al tomar una tendencia suavizada en lugar de reaccionar a los gradientes en crudo y acelera la convergencia con clases múltiples. 

### Scheduler StepLR (step_size=8, gamma=0.5)
Cada 8 épocas StepLR reduce el learning rate a la mitad (de la 1-8 --> lr = 3e-4, de la 9-16 --> lr = 1.5e-4, así sucesivamente), lo que ayuda a aprender rápido al principio y afinar pesos con pasos cada vez más pqueños evitando que el modelo se quede rebotando alrededor del mínimo. 

### CrossEntropy
Es la función de pérdida más utilizada para clasificación multiclase y mide la diferencia entre la distribución predicha y la verdadera; penaliza las predicciones incorrectas proporcionalmente a su nivel de confianza.

## Transformaciones
Se aplicaron transformaciones aleatorias tipo ImageNet para aumentar la robustez del modelo
- **Resize y RandomResizedCrop** para cambiar el encuadre de la imagen y simular variaciones de zoom.
- **RandomHorizontalFlip** para reconocer peces sin importar su dirección.
- **RandomRotation** para simular variaciones de orientación comunes en fotografías subacuáticas o de laboratorio.
- **ColorJitter** para manejar diferencias de iluminación entre imágenes modificando brillo, contraste y tono.
- **Normalización tipo ImageNet** para estabilizar el rango de entrada y mejorar convergencia.

Las augmentaciones ampliaron artificialmente el dataset, el cual contiene fotos con variaciones reales de luz, enfoque, tamaño, perspectiva etc --> al transformar, se logró entrenar un modelo más generalizable que no depende de condiciones/features específicas. 

---

# Iteración 1

## Entrenamiento

- **epochs**: 5
- **batch_size** = 32
- **num_workers** = 0 (carga de datos secuencial) 
- **Optimizador**: AdamW con lr = 3e-4 y weight_decay = 1e-4 
- **Scheduler StepLR**: step_size = 8, gamma = 0.5 (en esta iteración no llega a activarse porque solo hay 5 épocas)
- **Función de pérdida**: CrossEntropyLoss (label_smoothing = 0.0)

Se definieron esos parámetros como baseline para tener una configuración relativamente segura y estándar para un primer experimento. 

- Pocas ápocas para un chequeo rápido del pipeline de (modelo, dataloaders, transformaciones, optimizador) sin invertir demasiado tiempo de ejecución; el objetivo no era maximizar el rendimiento sino verificar que todo funcionara correctamente y que el modelo empezara a aprender algo útil. 
- batch_size = 32 es un valor típico para equilibrar estabilidad del gradiente y uso de memoria --> suficientemente grande para estimar bien el gradiente pero no tanto como para romper la RAM. 
- num_workers = 0 en Windows es lo más estable para una primera iteración porque evita problemas de multiproceso al cargar imágenes. 
- AdamW es bueno para clasificadores de imágenes porque combina la adaptatividad de Adam con regularización por weight decay desacoplada; el learning rate 3e-4 es conservador (no muy grande ni muy pequeño) y el weight decay ayuda a evitar que los pesos crezcan demasiado y se sobreajuste desde la primera iteración.
- Scheduler StepLR (step_size = 8, gamma = 0.5) en esta iteración no se activa pero se deja configurado desde el inicio para mantener la misma receta de entrenamiento en todas las iteraciones y aumentar las épocas sin cambiar la lógica del código.
- CrossEntropyLoss (label_smoothing = 0.0) es la función de pérdida estándar para clasificación multiclase, en el baseline se usa sin label smoothing para tener una referencia limpia del desempeño de la arquitectura sin transformaciones extra.

## Resultados en val
**loss**: 1.6844, **acc**: 0.5162, **F1**: 0.4318

El modelo sí está aprendiendo algo, no está al nivel de azar pero todavía está underfitted pues 5 épocas no son suficientes para que la red aproveche bien todo el dataset; el acc: 0.5162 y F1: 0.4318 indican que hay clases que ya se reconocen decentemente pero el rendimiento promedio por clase sigue siendo limitado (algunas clases tienen peor desempeño que otras).

Estos resultados sirven como baseline y en las siguientes iteraciones se aumentarán épocas, workers, early stopping etc, y se van a comparar contra este punto para justificar las mejoras en rendimiento.

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
- **patience**: 7 **(early stopping)**
- **batch_size** = 32
- **num_workers** = 4 **(data loader más rápido)** 
- **Optimizador**: AdamW con lr = 3e-4 y weight_decay = 1e-4 
- **Scheduler StepLR**: step_size = 8, gamma = 0.5
- **Función de pérdida**: CrossEntropyLoss (label_smoothing = 0.0)

## Resultados en val
**loss**: 0.4195, **acc**: 0.8902, **F1**: 0.8750

**Mejor época**: epoch 25 (el entrenamiento llegó hasta el final y no se hizo early stopping).

## Comparativa
Durante la primera iteración, el modelo sólo entrenó 5 épocas y terminó con acc: 0.5162 y F1: 0.4318, indicando un modelo subentrenado pues concluyó el entrenamiento antes de que la red comenzara a aprender patrones. En esta iteración se aumentaron las épocas de 5 a 25 dándole al modelo más tiempo para ajustar todos los pesos de la red convolucional, también se agregó early stopping con patience: 7 para evitar overfitting pero no se activó en esta iteración (nunca se detuvo antes). Igual se aumetó num_workers a 4 para agilizar la lectura de imágenes sin afectar la calidad del modelo. 

En validación el accuracy subió de 0.3166 al inicio hasta 0.8902, lo que valida que la arquitectura definida tiene la suficiente capacidad pero necesitaba más pasos de optimización, el F1: 0.875 indica que el modelo no sólo acierta globlamente sino que tiene un buen rendimiento en casi todas las clases, aún en las menos frecuentes; y el loss bajó de 2.413 a 0.4195 lo que significa que el modelo tiene más aciertos y la confianza es mayor en la clase correcta.
Lo que explica esta mejora de desempeño del modelo es que ahora se entrenó con más épocas y más tiempo de manera estable, manteniendo las demás configuraciones como estaban inicialmente. 

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

No hay ninguna clase que esté totalmente sin aciertos, confirmando que el modelo aprende todas las clases aunque algunas sean más difíciles que otras, alineado con el F1: 0.875 --> el modelo no solo tiene buena accuracy global, sino que también se comporta bien por clase. 

### Matriz de confusión Test

<img width="423" height="434" alt="cm_test2" src="https://github.com/user-attachments/assets/39bb3ea2-34ce-4d02-9485-52401079e4f9" />

El patrón de esta matriz es muy similar al de val con diagonales fuertes y errores concentrados en pocas clases, el accuracy en test de aprox 0.88 está muy cerca de la de val: 0.8902 --> el modelo no se sobreajusta al set de val sino que generaliza de verdad. 

Las clases fuertes (Glass Perchlet, Grass Carp, Gourami, Tilapia...) mantienen muchos aciertos, y clases más difíciles (Mudfish, Climbing Perch, Tenpounder...) presentan algunas confusiones pero aun así con un número decente de aciertos.

El comportamiento en test confirma que el modelo de esta segunda iteración ya es un clasificador útil y estable en lugar de sólo un experimento en train/val.

---

# Iteración 3 

## Aspectos de implementación 
Desde la parte de implementación, esta iteración además de ser la mejor fue también la más costosa al no lograr utilizar la GPU; se intentó CUDA y después DirectML, pero la combinación con AdamW no era estable y aparecieron problemas con algunas operaciones no soportadas y backprop, por lo que el entrenamiento se ejecutó completamente en CPU. En la anterior iteración se usaron 4 num_workers pero en esta definitiva se optó por usar 0 para evitar problemas en Windows con el dataloader y maximizar la estabilidad. Todo eso implicó que el entrenamiento de las 50 épocas tomara alrededor de 20 horas de ejecución contínua, pero resultó en un modelo mucho más solido y confiable.

## Entrenamiento
- **epochs**: 50
- **patience**: 7
- **batch_size** = 32
- **num_workers** = 0 **(evitar errores en Windows)** 
- **Optimizador**: AdamW con lr = 3e-4 y weight_decay = 1e-4 
- **Scheduler StepLR**: step_size = 8, gamma = 0.5 
- **Función de pérdida**: CrossEntropyLoss (label_smoothing = 0.0)

## Resultados
**Validation** 
- **loss**: 0.2391, **acc**: 0.9469, **F1**: 0.9370
- **Mejor época**: epoch 49 

**Test**
- **loss**: 0.2763, **acc**: 0.9330, **F1**: 0.9216

## Gráficas

### Loss por época
<img width="452" height="361" alt="loss" src="https://github.com/user-attachments/assets/0459bbd2-776c-48f7-893d-9af614cc7666" />

**Primeras 10 épocas** 
- train loss = baja de aprox 2.80 a 1.15
- val loss = baja de aprox 2.50 a 1.04

El modelo deja de casi adivinar a capturar patrones claros discriminando bien las 31 clases en lugar de hacerlo al azar. 

**Épocas 10 - 25**
- train loss = baja hasta aprox 0.48
- val loss = baja hasta aprox 0.39 - 0.40

La curva de val muestra oscilaciones con algunas subidas puntuales pero tiende a ser descendiente. El modelo necesita más ciclos para aprovechar toda la capacidad de la arquitectura y de los datos. 

**Épocas 25 - 50**
- train loss = baja de aprox 0.48 a 0.29
- val loss = baja de aprox 0.39 a 0.24 --> con algunos picos pero siempre con tendencia decreciente. 

En las últimas 10 épocas el loss de val se estabiliza en un rango muy bajo y en la época 49 se alcanza el mínimo global de 0.239; que el loss de val sea un poco menor que el de train en el tramo final indica que no hay overfitting fuerte, la regularización de dropout y weight decay sí funciona, el aprendizaje es progresivo y estable y el modelo generaliza muy bien.  

### Accuracy por época
<img width="452" height="361" alt="acc" src="https://github.com/user-attachments/assets/28f30c53-a1e6-48f2-9146-c71fc6abae78" />

**Primeras 10 épocas** 
- train acc = sube de aprox 0.23 a 0.67
- val acc = sube de aprox 0.31 a 0.70

Igual que con el loss se ve una fase de aprendizaje rápido donde cada época aporta mejoras significativas.

**Épocas 10 - 25**
- train acc = sube de aprox 0.67 a 0.87
- val acc = sube de aprox 0.70 a 0.90

Val se mantiene siempre muy cerca de train, sino que ligeramente arriba, lo cual es señal de buena calibración del modelo. 

**Épocas 25 - 50**
- train acc = sube de aprox 0.87 a 0.94
- val acc = sube de aprox 0.90 al máximo en la época 49 (0.9469).

La brecha entre train y val se mantiene pequeña; no hay un crecimiento de acc en train acompañado de una caída en val, por lo que el modelo no se memoriza el set de train. La red sigue ganando precisión prácticamente durante toda la ventana de 50 épocas, lo que explica por qué al comparar con la segunda iteración el accuracy de val subió de aprox 0.89 a 0.9469 --> necesitaba más tiempo para converger.

### Matriz de confusión Validation
<img width="423" height="434" alt="cm_val" src="https://github.com/user-attachments/assets/e612f851-4fe1-4bc3-b3f2-92e5903bc417" />

En la matriz se observa que la diagnoal está muy cargada con muchos valores cercanos o iguales al total de muestras de cada clase con recall casi perfecto, lo cual justifica el F1: 0.937 --> la mayoría de las clases tienen buen equilibro entre precisión y recall, sin embargo las clases más difíciles siguen siendo similares a las de iteraciones anteriores, por ejemplo las clases Mudfish, Bangus y/o Tenpounder muestran algunos errores de confusión cone species que se parecen visualmente pero aún así la red mantiene recalls razonables, además de que el número de errores por clase es bajo. 

Comparado a las otras dos iteraciones las confusiones se redujeron considerablemente tanto en número como en dispersión, la diagonal es más limpia y los valores fuera de ella son mucho más pequeños, o sea que el modelo discrimina mejor entre cada especie de pez. 

### Matriz de confusión Test
<img width="423" height="434" alt="cm_test" src="https://github.com/user-attachments/assets/59aaa0d0-772d-4a3a-a3cb-eeff80c3fbbf" />

Esta matriz mantiene el mismo patrón de diagonal visible y pocas confusiones recurrentes, confirma que el modelo generaliza a datos completamente no vistos, no hay ninguna clase con desempeño muy bajo (pocos aciertos), el modelo es consistente y los errores que persisten son razonables considerando el número de clases y la variabilidad visual de las especies de peces incluídas en el dataset. 

## Mejoras

Aunque no cambió la arquitectura base, las mejoras en esta iteración se deben primeramente al aumento de épocas porque el modelo necesitaba más tiempo para explotar su capacidad; de igual manera se llevó a cabo un uso consistente del mismo esquema de optimización con AdamW (lr=3e-4, weight_decay=1e-4) y Scheduler StepLR (step_size=8, gamma=0.5) que reduce el learning rate de forma suave a medida que avanzan las épocas, lo que permite una fase inicial de exploración más agresiva y una fase final con pasos pequeños para refinamiento --> el modelo sigue mejorando sin perder estabilidad cuando se alarga el entrenamiento. 
La regularización fue moderada pero suficiente pues el dropout al final del modelo y weight decay ayudaron a que no hubiera overfitting evidente incluso con 50 épocas. 

---

## Visualización de resultados

Capturas de pantalla de la interfaz gráfica corriendo mientras lleva a cabo algunas clasificaciones de prueba. 

### 1. Knifefish

<img width="1919" height="345" alt="image" src="https://github.com/user-attachments/assets/279ca341-ae3d-4a3a-970a-aecefd4f44d8" />

Lo reconoció correctamente. 

### 2. Gourami

<img width="1919" height="450" alt="gourami" src="https://github.com/user-attachments/assets/7296632d-0c84-4129-94ae-aff99a28fcbc" />

Lo reconoció correctamente. 

### 3. Freshwater Eel

<img width="1919" height="366" alt="eel" src="https://github.com/user-attachments/assets/97231ebe-1099-4ad0-96a9-0b69fee4cc2f" />

Lo reconoció correctamente. 

### Clases difíciles

### 1. Tenpounder 

<img width="1919" height="507" alt="tenpounder" src="https://github.com/user-attachments/assets/11a3239f-a196-481e-bfc1-d4fdf57adab6" />

Aquí se ve como en una de las clases más difíciles (Tenpounder) el modelo confunde a este pez con un Indo-Pacific Tarpon, el cual se ve así: 

<img width="259" height="194" alt="tarpon" src="https://github.com/user-attachments/assets/ce6f980d-052c-4893-9cf5-8f4770bbe504" /> 

Se puede apreciar un poco el parecido en la morfología de los peces ergo la confusión del modelo. 

### 2. Mudfish

<img width="1919" height="863" alt="mudfish" src="https://github.com/user-attachments/assets/af0b43ed-9f69-41b3-bc3b-3250a36ed91a" />

Lo reconoció correctamente pero con menos confianza que en los casos anteriores. 

### 3. Climbing Perch

<img width="1919" height="866" alt="climbingperch" src="https://github.com/user-attachments/assets/73496bd6-45e1-4a01-8a64-511d70e425df" />

Lo reconoció correctamente. 

### 4. Imágenes completamente no vistas 

<img width="1919" height="504" alt="bangusweb" src="https://github.com/user-attachments/assets/b14b8219-5b78-476f-a9a4-2924d14bedbb" />

Esta es una imagen de un pez Bangus sacada de internet, el modelo lo identificó correctamente. 

<img width="1919" height="484" alt="koiweb" src="https://github.com/user-attachments/assets/12d4691a-36d3-44f4-9d35-e449462c7057" />

Esta es una imagen de peces koi sacada de internet, el modelo los confundió con peces dorados, pues no entrenó con ningún ejemplo de pez koi. 

---

# Conclusiones

Se logró entrenar un modelo de clasificación de peces basado en una CNN profunda aplicando correctamente las etapas clave del aprendizaje profundo que son preprocesamiento, feature extraction, optimización y evaluación con las métricas reales. Por medio de tres iteraciones se observó que aumentar el número de épocas y mantener un entrenamiento estable con AdamW, StepLR, Dropout, augmentaciones de datos etc mejoró significativamente la capacidad de generalización del modelo hasta alcanzar un 94.69% de accuracy en validación. 

Aunque el entrenamiento se debió realizar en GPU el proceso confirmó el impacto directo de las decisiones tomadas con respecto a mayor profundidad efectiva, regularización adecuada y aprendizaje más prolongado, las cuales resultaron en mejores representaciones internas y clasificación más precisa. 

--- 

# Referencias

- Valdés Aguirre, B. (2025). _Módulo 2: Técnicas y arquitecturas de Deep Learning_. [Manuscrito no publicado]. Google Docs. https://docs.google.com/document/d/10VVnjkQejnhKR2ExMC4IsFVDdEgZ1Q6a/edit
- Olu-Ipinlaye, O., & Mukherjee, S. (2025, April 29). _A guide to global pooling in neural networks_. DigitalOcean Community Tutorials. https://www.digitalocean.com/community/tutorials/global-pooling-in-convolutional-neural-networks
- Yassin, A. (2024, November 8). _Adam vs. AdamW: Understanding weight decay and its impact on model performance_. Medium. https://yassin01.medium.com/adam-vs-adamw-understanding-weight-decay-and-its-impact-on-model-performance-b7414f0af8a1

---

<img  src="https://github.com/user-attachments/assets/acbb9132-e825-4a4a-8962-da8e1cede0cb" alt="descarga" width="452" height="361" /> 
