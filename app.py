import streamlit as st
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import random
import os
from collections import Counter

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Load model and preprocessing objects
def load_resources():
    model = keras.models.load_model('chat_model.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    return model, tokenizer, lbl_encoder

def train_and_save():
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, BatchNormalization
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.preprocessing import LabelEncoder
    import pickle
    from sklearn.model_selection import train_test_split
    from collections import Counter

    with open('intents.json') as file:
        data = json.load(file)
    
    training_sentences = []
    training_labels = []
    labels = []
    responses = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern.lower())  # Convert to lowercase
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    num_classes = len(labels)

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels_enc = lbl_encoder.transform(training_labels)

    # Increased vocabulary size and max length
    vocab_size = 2000
    embedding_dim = 64  # Increased embedding dimension
    max_len = 30  # Increased max length
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(training_sentences)
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

    # Check class distribution before splitting
    class_counts = Counter(training_labels_enc)
    min_class_count = min(class_counts.values())
    # Optional: print class distribution for debugging
    print("Class distribution:", class_counts)

    if min_class_count < 2:
        # If any class has less than 2 samples, do not stratify
        X_train, X_val, y_train, y_val = train_test_split(
            padded_sequences, np.array(training_labels_enc),
            test_size=0.2, random_state=42
        )
    else:
        # Stratify if possible
        X_train, X_val, y_train, y_val = train_test_split(
            padded_sequences, np.array(training_labels_enc),
            test_size=0.2, random_state=42, stratify=training_labels_enc
        )

    # Improved model architecture
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Add early stopping and reduce learning rate on plateau
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001
    )

    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=8,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Save model with .keras extension for compatibility
    model.save("chat_model.keras")
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return history

# Add a button to train the model
if st.button('Train Model'):
    with st.spinner('Training model... This may take a few minutes.'):
        history = train_and_save()
    st.success('Model trained and saved! Please reload the app to chat with the new model.')
    
    # Show training results
    if history:
        st.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        st.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Only load resources if model exists
if os.path.exists('chat_model.keras') and os.path.exists('tokenizer.pickle') and os.path.exists('label_encoder.pickle'):
    model, tokenizer, lbl_encoder = load_resources()
    max_len = 30  # Updated to match training
else:
    model = tokenizer = lbl_encoder = None
    max_len = 30

st.title("🤖 ML Chatbot")
st.write("Type your message below and press Enter. Type 'quit' to stop.")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if model and tokenizer and lbl_encoder:
    user_input = st.text_input("You:", "")
    if user_input:
        if user_input.lower() == "quit":
            st.write("Chat ended.")
        else:
            # Preprocess input
            processed_input = user_input.lower().strip()
            sequence = tokenizer.texts_to_sequences([processed_input])
            
            # Check if sequence is empty (word not in vocabulary)
            if not sequence or all(s == 0 for s in sequence[0]):
                response = "I'm not familiar with those words. Could you try rephrasing?"
            else:
                padded_sequence = pad_sequences(sequence, truncating='post', maxlen=max_len)
                
                result = model.predict(padded_sequence, verbose=0)
                predicted_prob = np.max(result)
                
                # Add confidence threshold
                confidence_threshold = 0.7
                if predicted_prob < confidence_threshold:
                    response = "I'm not sure I understand. Could you rephrase that?"
                else:
                    tag = lbl_encoder.inverse_transform([np.argmax(result)])
                    response = None
                    for i in data['intents']:
                        if i['tag'] == tag[0]:
                            response = random.choice(i['responses'])
                            break
                    if response is None:
                        response = "Sorry, I didn't understand that."
            
            st.session_state['chat_history'].append((user_input, response))
            
            # Clear the input after processing
            st.experimental_rerun()
else:
    st.warning('Model not found. Please train the model first.')

for user, bot in st.session_state['chat_history']:
    st.markdown(f"**You:** {user}")
    st.markdown(f"**ChatBot:** {bot}")
