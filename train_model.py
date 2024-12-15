import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Cargar el dataset (reemplazar 'ruta_dataset' con la ruta real)
dataset = pd.read_csv('dataset_comentarios.csv', encoding='latin-1')
comentarios = dataset['comentario'].values
sentimientos = dataset['sentimiento'].values

# Convertir etiquetas de texto a valores numéricos (positivo: 1, negativo: 0)
sentimientos = np.array([1 if s == 'positivo' else 0 for s in sentimientos])

# Preprocesamiento del texto
vocab_size = 40000  # Tamaño del vocabulario
max_length = 300    # Longitud máxima de cada comentario
oov_tok = "<OOV>"   # Token para palabras desconocidas

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(comentarios)
sequences = tokenizer.texts_to_sequences(comentarios)
padded_sequences = pad_sequences(sequences, maxlen=max_length, truncating='post')

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, sentimientos, test_size=0.2, random_state=42)

# Definir la estructura de la red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 100, input_length=max_length),
    tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Salida binaria
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=2
)

# Guardar el modelo entrenado
model.save('sentimento_model.h5')
print("Modelo guardado como 'sentiment0_model.h5'")


# Guardar el tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Tokenizador guardado como 'tokenizer.pkl'")
