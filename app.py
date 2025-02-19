from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st 
import pickle
import numpy as np 


#Load the LSTM Model

model=load_model('next_word_gru.h5')

# Load tokenizer

with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model,tokenizer,text,max_sequence):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list) > max_sequence:
        token_list=token_list[-(max_sequence-1):]
    token_list=pad_sequences([token_list],maxlen=max_sequence-1,padding='pre')
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)
    for w,i in tokenizer.word_index.items():
        if i==predicted_word_index:
            return w
    return None
#Streamlit app
st.title('Next Word Prediction using LSTM')

input_text=st.text_input("Enter the setnence","To be or not")
if st.button("Predict next word"):
    max_sequence=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence)
    st.write(f"Next word: {next_word}")

