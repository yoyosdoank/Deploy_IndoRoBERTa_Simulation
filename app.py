import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter
import os
from langdetect import detect, LangDetectException

def has_vowel(word):
    vowels = 'aeiouAEIOU'
    return any(char in vowels for char in word)

def has_consecutive_letters(word):
    count = 1
    for i in range(1, len(word)):
        if word[i] == word[i - 1]:
            count += 1
            if count >= 3:
                return True
        else:
            count = 1
    return False

@st.cache_resource()
def get_model():
    tokenizer = AutoTokenizer.from_pretrained('flax-community/indonesian-roberta-base')
    model1 = AutoModelForSequenceClassification.from_pretrained("yogie27/IndoRoBERTa-Sentiment-Classifier-for-Twitter", token="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")
    model2 = AutoModelForSequenceClassification.from_pretrained("yogie27/IndoRoBERTa-Emotion-Classifier-for-Twitter", token="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")
    return tokenizer,model1,model2

tokenizer,model1,model2 = get_model()

st.image("banner_edit.png", use_column_width=True)
header = st.title("Prediksi Sentimen & Emosi Untuk Media Sosial Berbasis Teks Bahasa Indonesia Dengan Metode Transformers.")
note1 = st.caption("**Author: Yogie Oktavianus Sihombing**")
note2 = st.write("SENTIMEN adalah sikap, perasaan, atau pandangan yang lebih stabil dan cenderung bertahan lebih lama terhadap seseorang, situasi, atau fenomena tertentu. Sentimen merupakan cerminan dari emosi yang lebih menetap dan terinternalisasi. EMOSI adalah respons psikologis yang intens, sering kali singkat, terhadap suatu peristiwa atau situasi. Emosi biasanya bersifat sementara dan bisa berubah dengan cepat. ***- Ivanov, D. (2023) -***")
st.info("Masukkan kalimat Anda di kolom bawah dan tekan 'ANALISIS' untuk mulai prediksi. Tekan 'RESET' untuk atur ulang halaman.")


sentimen = {
  2:'Positif',  
  1:'Netral',
  0:'Negatif'
}

emosi = {
  4:'Sedih - Kecewa',
  3:'Sayang',
  2:'Senang - Bahagia',  
  1:'Takut - Khawatir',
  0:'Marah - Jijik'
}

# Membuat form
user_input = st.text_area('**MASUKKAN KALIMAT:**')
with st.form(key='my_form'):
    button = st.form_submit_button("ANALISIS")
    reset_button = st.form_submit_button("RESET")

    if button and user_input:
        if len(user_input.split()) > 7:
            english_word_count = 0
            indonesian_word_count = 0
            warning_count = 0
            for word in user_input.split():
                try:
                    lang = detect(word)
                    if lang == 'en':
                        english_word_count += 1
                    elif lang == 'id':
                        indonesian_word_count += 1
                    if has_consecutive_letters(word):
                        st.warning(f"Kata '{word}' memiliki tiga atau lebih huruf yang sama berurutan, dapat mempengaruhi konteks dan prediksi.")
                        warning_count += 1
                    if not has_vowel(word):
                        st.warning(f"Kata '{word}' tidak memiliki huruf vokal, dapat mempengaruhi konteks dan prediksi.")
                        warning_count += 1
                except LangDetectException:
                    st.warning(f"Tidak dapat mendeteksi bahasa untuk kata '{word}'.")
                    warning_count += 1

            if warning_count > 5:
                st.error("Warning lebih dari 5, analisis prediksi dihentikan, perbaiki kembali kalimat Anda.")
            else:
                if english_word_count > indonesian_word_count:
                    st.warning("Kalimat ini dominan dalam bahasa Inggris, dapat mempengaruhi konteks dan prediksi.")

                inputs = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')

                output1 = model1(**inputs)
                output2 = model2(**inputs)

                logits1 = output1.logits
                logits2 = output2.logits

                max_sentiment_index = torch.argmax(logits1, dim=1).item()
                max_sentiment_prob = torch.softmax(logits1, dim=1).squeeze()[max_sentiment_index].item()

                max_emotion_index = torch.argmax(logits2, dim=1).item()
                max_emotion_prob = torch.softmax(logits2, dim=1).squeeze()[max_emotion_index].item()

                st.write("Sentimen:", f"**{sentimen[max_sentiment_index]}**", "; Persentase Prediksi:", f"**{max_sentiment_prob:.2%}**")
                st.write("Emosi:", f"**{emosi[max_emotion_index]}**", "; Persentase Prediksi:", f"**{max_emotion_prob:.2%}**")
        else:
            st.error("Panjang 1 kalimat disarankan lebih dari 7 kata untuk memahami konteks dalam kalimat, input kembali pada kolom teks.")

    if reset_button:
        st.experimental_rerun()

st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: green;
        color: white;
    }
    div.stButton > button:last-child {
        background-color: red;
        color: white;
    }
    .stForm {
        border: 2px solid blue;
        padding: 1em;
    }
    </style>
    """, unsafe_allow_html=True)

# Menggabungkan beberapa catatan menjadi satu kotak informasi dengan latar belakang biru
st.markdown("""
<div style="background-color: #d1ecf1; padding: 10px; border-radius: 5px; border: 1px solid #bee5eb;">
    <p><strong>Harap memasukkan kalimat yang mempunyai konteks, minimal 7 kata dalam 1 kalimat.</strong></p>
    <p><strong>Rekomendasi media sosial berbasis teks: Twitter.</strong></p>
    <p><strong>Dimungkinkan analisis dari media sosial lainnya.</strong></p>
    <p><strong>Analisis selain menggunakan bahasa Indonesia tidak dibenarkan.</strong></p>
</div>
""", unsafe_allow_html=True)
