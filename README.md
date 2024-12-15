# Análisis de Sentimientos con Redes Neuronales LSTM
**Autor:** Nelson Mamani Ramos  
**Materia:** Inteligencia Artificial II

## Descripción del Proyecto
Este proyecto utiliza redes neuronales LSTM (Long Short-Term Memory) para realizar análisis de sentimientos sobre un conjunto de comentarios. El modelo clasifica los comentarios en dos categorías: positivos y negativos. Se ha entrenado utilizando un conjunto de datos de comentarios de texto y evaluado en base a su capacidad para predecir el sentimiento de nuevos comentarios.

El proyecto está implementado en Python utilizando TensorFlow, y se entrena sobre un dataset proporcionado para realizar la tarea de clasificación de sentimientos. El modelo ha sido guardado y se puede utilizar para predecir el sentimiento de nuevos comentarios.

## Requisitos
Antes de ejecutar el proyecto, asegúrate de tener instalado Python 3.x. Puedes instalar todos los paquetes necesarios utilizando un entorno virtual.

### Instalación de Dependencias:
1. **Crear un entorno virtual:**

    Si no tienes un entorno virtual creado, puedes crear uno con el siguiente comando:
    ```bash
    python -m venv venv
    ```

2. **Activar el entorno virtual:**

    En Windows:
    ```bash
    .\venv\Scripts\activate
    ```

    En macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

3. **Instalar los requisitos:**
    Con el entorno virtual activado, instala las dependencias necesarias ejecutando:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset
El dataset utilizado para entrenar el modelo no está incluido directamente en el repositorio debido a su tamaño. Puedes descargarlo desde el siguiente enlace:

[Descargar dataset_comentarios.csv (70mb)](https://www.mediafire.com/file/8dpjxrxbvc7w68u/dataset_comentarios.csv/file)

Una vez descargado, coloca el archivo `dataset_comentarios.csv` en la misma carpeta donde se encuentra el script principal del proyecto.

## Ejecución del Proyecto
(si solo quieres ejecutar el modelo ya entrendao salta al paso 3, entrenar el modelo puede llevarte mas de 30 minutos como en mi caso)

Para ejecutar el proyecto y utilizar el modelo entrenado:

1. **Asegúrate de tener el archivo `dataset_comentarios.csv` en el directorio del proyecto**.
   
2. **Ejecutar el entrenamiento (si aún no se ha realizado)**:
    Si aún no has entrenado el modelo, puedes ejecutar el script `train_model.py` para entrenar el modelo utilizando el dataset descargado:
    ```bash
    python entrenar_modelo.py
    ```

    Esto entrenará el modelo y guardará el archivo `sentimento_model.h5` (modelo entrenado) y `tokenizer.pkl` (tokenizador) en el directorio actual.

3. **Realizar predicciones**:
    Una vez que el modelo esté entrenado, puedes ejecutar el script `predict_sentiment.py` para hacer predicciones sobre nuevos comentarios de texto. Aquí está el ejemplo básico de cómo ejecutarlo:
    ```bash
    python probar_modelo.py
    ```

    El script tomará los comentarios y devolverá las predicciones sobre si el sentimiento es positivo o negativo.

## Estructura del Proyecto

- `entrenar_modelo.py`: Script para entrenar el modelo.
- `probar_modelo.py`: Script para realizar predicciones con el modelo entrenado.
- `sentimento_model.h5`: El modelo de red neuronal entrenado.
- `tokenizer.pkl`: Tokenizador utilizado para convertir los comentarios en secuencias de texto.
- `dataset_comentarios.csv`: Dataset utilizado para entrenar el modelo (descargable desde el enlace provisto).

## Licencia y uso
Este proyecto abierto para que puedas usarlo. Comparte abiertamente!
@nelson1mr

