# FINANCIAL FRAUD DETECTION

## PROBLEMA DE NEGOCIO

La urgencia por detectar fraudes en transacciones móviles de dinero ha llevado a una empresa del segmento Fintech a buscar soluciones innovadoras. Como científicos de datos hemos sido convocados para desarrollar un modelo de machine learning que pueda distinguir de manera precisa entre transacciones legítimas y fraudulentas, estableciendo así un estándar de seguridad en el sector financiero móvil global.

## DESCRIPCION DEL PROYECTO

Este proyecto utiliza técnicas de Machine Learning para detectar transacciones fraudulentas en un conjunto de datos. El objetivo es identificar patrones que puedan distinguir entre transacciones legítimas y fraudulentas.

## Herramientas utilizadas
- Colab
- Python
- Librerías:
  - Pandas
  - Numpy
  - Time
  - Matplotlib
  - StringIO
  - Sklearn
  - Imblearn
  - Warnings

## Configuración del Ambiente
Se realizó el trabajo en google Colab con los siguientes pasos:
- Configuración del Ambiente
- Obtención y tratamiento de datos

El archivo Financial_Fraud_Detection_BigData.ipynb es el código escrito en colab el cual contiene todas las operaciones de analisis, limpieza de datos, Insights construcción de modelos y evaluación.
Los datos base fueron extraidos de fuentes oficiales, https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset

## Pasos Desarrollados
![image](https://github.com/TigerXHero/Financial-Fraud-Detection/blob/main/images/pasos.png)

### 1. Preprocesamiento de Datos
En éste apartado se realizó:
- Limpieza de datos, manejo de valores faltantes, nulos, etc.
- Se realizó la adecuación y normalización de datosy tipo de datos obtenidos de la fuente de origen.
- Se implementó feature engineering para una mejor construcción de modelos de machine learning.

### 2. Exploración de Datos

Exploremos los datos del dataframe mediante visualizaciones

![image](https://github.com/TigerXHero/Financial-Fraud-Detection/blob/main/images/distribucion_fraudes.png)
Distribución de Fraudes

El gráfico muestra la distribución de transacciones fraudulentas frente a no fraudulentas. 
La gran mayoría de las transacciones (99.87%) no son fraudulentas, mientras que solo el 0.13% corresponde a fraudes. 
Esta desbalanceada proporción resalta la necesidad de un modelo robusto para detectar fraudes de manera efectiva.

Existe un desbalanceo muy grande entre la variable objetivo, es necesario hacer un balanceo más adelante para obtener un mejor modelo.

![image](https://github.com/TigerXHero/Financial-Fraud-Detection/blob/main/images/boxplot_monto.png)
Boxplot de Monto

Este boxplot visualiza la distribución del monto de las transacciones. 
La mayoría de las transacciones tienen montos pequeños, con algunos valores atípicos que llegan hasta los 80 millones. 
La presencia de estos valores atípicos puede influir en la detección de fraudes y debe tenerse en cuenta al entrenar los modelos.

Existe demasiados outliers en los montos de las transacciones.

![image](https://github.com/TigerXHero/Financial-Fraud-Detection/blob/main/images/transacciones_x_destinatario.png)
![image](https://github.com/TigerXHero/Financial-Fraud-Detection/blob/main/images/transacciones_mes.png)
El gráfico muestra que el número total de transacciones varía a lo largo del mes, con un pico notable el día 1, donde se registran 521,000 transacciones. 
El día 2, el 6 al 17, son picos relativamente altos consecutivamente al día 1.
Los días restantes presentan un número de transacciones relativamente bajo, con una media de 40,000 transacciones.

El pico del día 1 podría deberse a diversos factores, como:

- Efecto fin de mes
- Promociones o eventos especiales
- Errores en la recopilación de datos

![image](https://github.com/TigerXHero/Financial-Fraud-Detection/blob/main/images/transacciones_fraude.png)
Porcentaje de transacciones fraudulentas por día

Se observan picos de fraude en los días 3, 5, 19, 27 y 31, con porcentajes de fraude que alcanzan el 4.5%, 2.1%, 2.5%, 3.1% y 100%, respectivamente. 
Los días restantes presentan un porcentaje de fraude relativamente bajo, con un promedio de 0.6%. 
Es importante destacar que el último día del período analizado muestra un 100% de fraude, lo que requiere un análisis más detallado.

La variación del porcentaje de transacciones fraudulentas a lo largo del período analizado puede deberse a diversos factores, como:

- Patrones de compra
- Métodos de fraude
- Factores externos

![image](https://github.com/TigerXHero/Financial-Fraud-Detection/blob/main/images/transacciones_rango_horario.png)

El gráfico muestra que el número total de transacciones varía a lo largo del día. 
Se observa un pico de transacciones en la tarde, 3.13 millones de transacciones. 
Por la noche una cantidad de 2.22 millones de transacciones, seguido con una cantidad mas baja de 0.75 millones por la mañana y 0.11 millones de transacciones por la madrugada.

La variación del número total de transacciones a lo largo del día puede deberse a diversos factores, como:

- Patrones de compra: Las horas con mayor número de transacciones podrían coincidir con las horas de mayor actividad comercial, como la hora del almuerzo o la tarde.
- Factores externos: Factores externos, como eventos o noticias, podrían estar influyendo en la actividad comercial.

![image](https://github.com/TigerXHero/Financial-Fraud-Detection/blob/main/images/mapa_calor_1.png)
El mapa de calor presentado muestra la correlación entre diferentes variables relacionadas con las transacciones. 
Las variables se presentan en una matriz, con cada celda representando la correlación entre dos variables específicas. 
El color de cada celda indica la fuerza de la correlación, siendo el rojo más intenso una correlación positiva fuerte y el azul más intenso una correlación negativa fuerte.

De acuerdo a las correlaciones observadas, el tipo de pago con mayor probabilidad de fraude es type_TRANSFER. 

Los métodos de pago en efectivo ("type_CASH_IN" y "type_CASH_OUT") y los métodos de pago con tarjeta ("type2_CC" y "type2_CM") están asociados con una menor probabilidad de fraude. 
Esto podría deberse a que estos métodos de pago son más difíciles de falsificar o utilizar para actividades fraudulentas.
Las transacciones de mayor monto tienden a tener una menor probabilidad de fraude. 
Esto podría deberse a que los defraudadores suelen realizar transacciones de menor monto para evitar llamar la atención.

### 3. Construcción de Modelos
![image](https://github.com/TigerXHero/Financial-Fraud-Detection/blob/main/images/matriz_confusion_1.png)
![image](https://github.com/TigerXHero/Financial-Fraud-Detection/blob/main/images/matriz_confusion_2.png)
Observamos que los dos mejores modelos con mejores métricas son Random Forest y Árbol de Decisión

### 4. Evaluación y Selección del Modelo
![image](https://github.com/TigerXHero/Financial-Fraud-Detection/blob/main/images/evaluacion_modelos.png)
Observamos que los dos mejores modelos con mejores métricas son **Random Forest** y **Árbol de Decisión**
## Conclusiones:

Determinamos que el modelo Random Forest era la opción más apropiada debido a sus ventajas específicas en términos de métricas de desempeño y capacidad de generalización en conjuntos de datos grandes, complejos y desequilibrados.

La implementación de este modelo en un entorno bancario puede mejorar significativamente la capacidad de detectar y prevenir fraudes, reduciendo así las pérdidas económicas y aumentando la confianza de los clientes en la seguridad de las transacciones.

## Autores

- [@Gerardo](https://github.com/GeraDLC)
- [@Isaias](https://github.com/TigerXHero)
- [@Cristobal]()
