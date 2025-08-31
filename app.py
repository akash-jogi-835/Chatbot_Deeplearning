import streamlit as st
import numpy as np
import json
from tensorflow import keras
import pickle
import random
import os

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Load model and preprocessing objects
def load_resources():
    model = keras.models.load_model('chat_model')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    return model, tokenizer, lbl_encoder

def train_and_save():
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.preprocessing import LabelEncoder
    import pickle

    with open('intents.json') as file:
        data = json.load(file)
    
    training_sentences = []
    training_labels = []
    labels = []
    responses = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    num_classes = len(labels)

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels_enc = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(padded_sequences, np.array(training_labels_enc), epochs=500)

    model.save("chat_model")
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

# Add a button to train the model
if st.button('Train Model'):
    train_and_save()
    st.success('Model trained and saved! Please reload the app to chat with the new model.')

# Only load resources if model exists
if os.path.exists('chat_model') and os.path.exists('tokenizer.pickle') and os.path.exists('label_encoder.pickle'):
    model, tokenizer, lbl_encoder = load_resources()
    max_len = 20
else:
    model = tokenizer = lbl_encoder = None
    max_len = 20

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
            result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]),
                                                 truncating='post', maxlen=max_len))
            tag = lbl_encoder.inverse_transform([np.argmax(result)])
            response = None
            for i in data['intents']:
                if i['tag'] == tag:
                    response = random.choice(i['responses'])
                    break
            if response is None:
                response = "Sorry, I didn't understand that."
            st.session_state['chat_history'].append((user_input, response))
else:
    st.warning('Model not found. Please train the model first.')

for user, bot in st.session_state['chat_history']:
    st.markdown(f"**You:** {user}")
    st.markdown(f"**ChatBot:** {bot}")
