# MODEL CODE FOR CADRS MODEL ADAPT TO SEQUENCE OUTPUT

model = tf.keras.Sequential([
    # Embedding layer
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    # LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # Dense layer
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # output layer
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
