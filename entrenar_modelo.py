import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo previamente guardado
model = load_model('sentimento_model.h5')
print("Modelo cargado exitosamente.")

# Cargar el tokenizer que se usó para entrenar el modelo
# Es necesario tener el tokenizador entrenado en el primer script para usarse aquí
# Para simplificar, cargamos el tokenizador que se guardó previamente como un ejemplo
import pickle
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Definir los comentarios nuevos
comentarios_nuevos = [
    "La película fue impresionante, realmente una obra maestra de cine.",  # positivo
    "El servicio en este restaurante fue lento y poco amable, no lo recomendaría.",  # negativo
    "La música de este álbum es única, nunca había escuchado algo tan innovador.",  # positivo
    "El libro tiene una trama interesante, pero los personajes son bastante planos y poco profundos.",  # negativo
    "Este café tiene el mejor ambiente para trabajar o relajarse, y la comida es deliciosa.",  # positivo
    "La serie tiene un buen comienzo, pero luego pierde ritmo y se vuelve predecible.",  # negativo
    "La actuación en la película fue fenomenal, especialmente el papel principal.",  # positivo
    "La comida del restaurante estaba fría y demasiado salada, no pienso volver.",  # negativo
    "El concierto fue increíble, aunque el lugar estaba demasiado lleno y no pudimos disfrutarlo al máximo.",  # negativo
    "Este lugar tiene una vista espectacular, ideal para relajarse y disfrutar del paisaje."  # positivo
]

# Preprocesar los nuevos comentarios
max_length = 300 
sequences_nuevos = tokenizer.texts_to_sequences(comentarios_nuevos)
padded_sequences_nuevos = pad_sequences(sequences_nuevos, maxlen=max_length, truncating='post')

# Hacer predicciones
predicciones = model.predict(padded_sequences_nuevos)

# Mostrar las predicciones
for i, comentario in enumerate(comentarios_nuevos):
    print(f"Comentario: {comentario}")
    print(f"Puntuación: {predicciones[i][0]} -> {'Positivo' if predicciones[i][0] > 0.5 else 'Negativo'}")
    print("------------------------------------------")

