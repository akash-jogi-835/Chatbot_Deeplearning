import streamlit as st
import numpy as np
import json
from tensorflow import keras
import pickle
import random
import os

# --- Sidebar ---
st.sidebar.title("🗂️ Navigation")
if 'all_chats' not in st.session_state:
    st.session_state['all_chats'] = []

if st.sidebar.button('New Chat'):
    if st.session_state.get('chat_history'):
        st.session_state['all_chats'].append(st.session_state['chat_history'])
    st.session_state['chat_history'] = []
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Chat History")
if st.session_state['all_chats']:
    for idx, chat in enumerate(st.session_state['all_chats']):
        if st.sidebar.button(f"Chat {idx+1}"):
            st.session_state['chat_history'] = chat
            st.experimental_rerun()
else:
    st.sidebar.write("No previous chats yet.")

# --- Main App ---
st.title("🤖 Jarvis - ML Chatbot")
st.markdown("<style>div.block-container{padding-top:2rem;} .stTextInput>div>div>input{font-size:1.1rem;} .stMarkdown{font-size:1.1rem;} .stButton>button{font-size:1.1rem;}</style>", unsafe_allow_html=True)
st.write("Type your message below and press Enter. Type 'quit' to stop.")

# Load intents
def load_intents():
    with open("intents.json") as file:
        return json.load(file)
data = load_intents()

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
    model.save("chat_model.keras")
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

if st.button('Train Model'):
    train_and_save()
    st.success('Model trained and saved! Please reload the app to chat with the new model.')

if os.path.exists('chat_model.keras') and os.path.exists('tokenizer.pickle') and os.path.exists('label_encoder.pickle'):
    model, tokenizer, lbl_encoder = load_resources()
    max_len = 20
else:
    model = tokenizer = lbl_encoder = None
    max_len = 20

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# --- Chat UI ---
chat_placeholder = st.container()

with chat_placeholder:
    if model and tokenizer and lbl_encoder:
        user_input = st.text_input("You:", "", key="user_input")
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
                st.experimental_rerun()
    else:
        st.warning('Model not found. Please train the model first.')

    # Display chat history in a styled format
    for user, bot in st.session_state['chat_history']:
        st.markdown(f"<div style='background-color:#e6f7ff;padding:8px;border-radius:8px;margin-bottom:4px;'><b>You:</b> {user}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#f0f0f0;padding:8px;border-radius:8px;margin-bottom:12px;'><b>Jarvis:</b> {bot}</div>", unsafe_allow_html=True)
