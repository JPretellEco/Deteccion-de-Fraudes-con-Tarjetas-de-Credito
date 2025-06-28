<html><head></head><body><h1>🛡️ Detección de Fraudes con Tarjetas de Crédito</h1>

<p>Este proyecto implementa un modelo de clasificación supervisada para detectar transacciones fraudulentas en tarjetas de crédito. Utiliza técnicas modernas de ingeniería de características, balanceo con SMOTE y un Árbol de Decisión como modelo base. La solución es modular y aplicable fácilmente a cualquier nuevo dataset de transacciones.</p>

<hr>

<h2>📂 Estructura del Proyecto</h2>

<p><code>
.
├── modelo.py                  # Entrenamiento y predicción del modelo
├── generador_features.py     # Preprocesamiento y limpieza de datos crudos
├── Flujo Preprocesamiento/   # Lógica completa de transformación (en Jupyter o script)
├── data/
│   ├── cruda/
│   │   ├── fraudTrain.csv
│   │   └── fraudTest.csv
│   └── procesada/
│       ├── data_procesada.csv
│       └── data_test.csv
├── README.md                 # Documentación del proyecto
</code></p>

<hr>

<h2>⚙️ Requisitos</h2>

<p>Instala las siguientes librerías antes de ejecutar el proyecto:</p>

<p><code>bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
</code></p>

<hr>

<h2>🚀 ¿Cómo ejecutar?</h2>

<h3>1. Preprocesamiento de Datos</h3>

<p>Ejecuta el script <code>generador_features.py</code>, que limpia y transforma los datos crudos (<code>fraudTrain.csv</code>, <code>fraudTest.csv</code>).</p>

<p><code>bash
python generador_features.py
</code></p>

<p>Esto generará:</p>

<ul>
<li><code>data_procesada.csv</code> para entrenamiento</li>
<li><code>data_test.csv</code> para predicciones</li>
</ul>

<p>Estos archivos se guardan en <code>data/procesada/</code>.</p>

<hr>

<h3>2. Entrenamiento y Evaluación del Modelo</h3>

<p>Ejecuta <code>modelo.py</code>:</p>

<p><code>bash
python modelo.py
</code></p>

<p>Esto realizará automáticamente:</p>

<ul>
<li>Carga de datos</li>
<li>Preprocesamiento (codificación)</li>
<li>Balanceo con SMOTE</li>
<li>Entrenamiento con Árbol de Decisión (<code>DecisionTreeClassifier</code>)</li>
<li>Evaluación con métricas clave</li>
<li>Visualización de:
<ul>
<li>Matriz de confusión</li>
<li>Curva ROC</li>
<li>Importancia de variables</li>
</ul></li>
<li>Predicciones sobre el dataset test (<code>data_test.csv</code>)</li>
</ul>

<hr>

<h2>🧪 Resultados esperados (ejemplo)</h2>

<ul>
<li><strong>Accuracy</strong>: 0.93  </li>
<li><strong>Recall (fraudes detectados)</strong>: 0.88  </li>
<li><strong>F1-score</strong>: 0.90  </li>
<li><strong>ROC AUC</strong>: 0.94  </li>
</ul>

<p>Estas métricas indican una buena capacidad del modelo para detectar fraudes, incluso con clases desbalanceadas.</p>

<hr>

<h2>🔍 Variables Clave del Modelo</h2>

<p>El modelo utiliza las siguientes variables después del feature engineering:</p>

<p>
| Variable                  | Descripción                                          |
|---------------------------|------------------------------------------------------|
| <code>Grupo_Edad</code>              | Agrupación de edad por rango                         |
| <code>Tipo_Via</code>                | Tipo de calle o ubicación comercial                  |
| <code>Horario</code>                 | Periodo del día (Mañana, Tarde, Noche, etc.)        |
| <code>Tipo_Tarjeta</code>            | Tipo estimado de tarjeta (Visa, Mastercard, etc.)   |
| <code>Categoria_Compra</code>        | Tipo de producto o servicio                          |
| <code>Trabajo</code>                 | Ocupación agrupada del cliente                      |
| <code>Genero</code>                  | Género del cliente (0 = F, 1 = M)                   |
| <code>Monto_Transaccion</code>       | Monto de la transacción                             |
| <code>Distancia_Cliente_Negocio</code> | Distancia entre cliente y punto de venta           |
</p>

<hr>

<h2>🔁 Funciones destacadas del Preprocesamiento</h2>

<p>El módulo <code>generador_features.py</code> transforma las variables a partir de:</p>

<ul>
<li>Extracción de hora y creación del periodo del día</li>
<li>Cálculo de la edad y agrupación en rangos</li>
<li>Distancia geográfica cliente-negocio (fórmula de Haversine)</li>
<li>Clasificación del tipo de tarjeta según los primeros dígitos</li>
<li>Agrupación de ocupaciones laborales</li>
<li>Limpieza y traducción de nombres de estados en EE.UU.</li>
<li>Clasificación del tipo de vía y tratamiento del monto</li>
</ul>

<hr>

<h2>🧩 Tecnologías y Algoritmos</h2>

<ul>
<li><strong>Python 3.x</strong></li>
<li><strong>SMOTE</strong>: para balancear clases</li>
<li><strong>Árbol de Decisión</strong> (<code>DecisionTreeClassifier</code>)</li>
<li><strong>ROC Curve + AUC</strong>: evaluación de clasificación binaria</li>
<li><strong>OneHot/Label/Ordinal Encoding</strong> según el tipo de variable</li>
</ul>

<hr>

<h2>📌 Posibles Mejoras</h2>

<ul>
<li>Probar modelos avanzados como XGBoost o LightGBM</li>
<li>Añadir tuning de hiperparámetros con <code>GridSearchCV</code></li>
<li>Aplicar <code>Pipeline</code> con <code>ColumnTransformer</code></li>
<li>Despliegue web con Flask o Streamlit</li>
<li>Registro en base de datos de resultados anómalos</li>
</ul>

<hr>

<h2>👨&zwj;💻 Autor</h2>

<p><strong>Jeffersson Pretell</strong><br>
Estudiante de Economía, Ciencia de Datos y Big Data<br>
Linuxero, apasionado por la programación y el análisis de datos<br>
🇵🇪 Perú</p>

<hr>

<h2>📬 Contacto</h2>

<p>¿Dudas o sugerencias? Puedes escribirme por GitHub o <a href="https://www.linkedin.com/in/">LinkedIn</a>.</p>

<hr>

<blockquote>
  <p><em>“Los datos son el nuevo petróleo, pero sin una refinería, solo tienes un pozo inútil.”</em></p>
</blockquote>
</body></html>
