# model.py
# Modelo de Deep Learning para Classificação de Texto usando TensorFlow e Keras
# Importações necessárias
# Autor: Pedro Falcão
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
# importações do TensorFlow e Keras
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense

# Desabilitar GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
load_dotenv()
# Conexão com o banco de dados
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Carregar dados do banco de dados
with engine.connect() as conn:
    # DataFrame
    df = pd.read_sql_query(text('SELECT * FROM "extrato";'), conn)
    # Checagem rápida de quantidade de linhas no banco
    total = conn.execute(text('SELECT COUNT(*) FROM "extrato";')).scalar()
    
# Selecionar colunas relevantes
df_filtered = df[['data', 'mes_ano', 'tipo_operacao', 'categoria', 'fornecedor', 'descricao', 'instituicao', 'categoria_financeira']].dropna()
# Concatenar colunas de texto para entrada
df_filtered['Texto'] = df_filtered['tipo_operacao'].astype(str) + " " + df_filtered['categoria'].astype(str) + " " + df_filtered['fornecedor'] + " " + df_filtered['descricao'] + " " + df_filtered['instituicao'].astype(str) + " " + df_filtered['categoria_financeira'].astype(str)
# Converter valores para string e normalizar
df_filtered['Texto'] = df_filtered['Texto'].astype(str).str.lower()

# Codificar a saída (categoria_financeira) para valores numéricos
label_encoder = LabelEncoder()
df_filtered['cat_fin_encoded'] = label_encoder.fit_transform(df_filtered['categoria_financeira'])

# Separar dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(df_filtered['Texto'], df_filtered['cat_fin_encoded'], test_size=0.2, random_state=42)

# Parâmetros do modelo
max_vocab_size = 5000
sequence_length = 100
embedding_dim = 16

# Tokenização e vetorização
vectorizer = TextVectorization(max_tokens=max_vocab_size, output_mode='int', output_sequence_length=sequence_length)
vectorizer.adapt(X_train.to_numpy())

# Converter os textos vetorizados para arrays numéricos
X_train = vectorizer(np.array(X_train)).numpy()
X_test = vectorizer(np.array(X_test)).numpy()

# Criar modelo de Deep Learning
model = keras.Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, mask_zero=True),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  # Saída com softmax para classificação
])

# Compilar modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Salvar o modelo treinado
model.save(f"modelo_classificacao.h5")
# Salvar o LabelEncoder
import pickle
with open(f"label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)