import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter
from langdetect import detect, LangDetectException
import re

# Definisi huruf vokal
def has_vowel(word):
    vowels = 'aeiouAEIOU'
    return any(char in vowels for char in word)

# Definisi huruf sama berurutan
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

# Definisi model deep learning
@st.cache_resource()
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")
    model1 = AutoModelForSequenceClassification.from_pretrained("yogie27/IndoRoBERTa-Sentiment-Classifier-for-Twitter", token="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")
    model2 = AutoModelForSequenceClassification.from_pretrained("yogie27/IndoRoBERTa-Emotion-Classifier-Base", token="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")
    return tokenizer, model1, model2

tokenizer, model1, model2 = get_model()

st.image("banner_edit.png", use_column_width=True)
header = st.title("Prediksi Sentimen & Emosi Untuk Media Sosial Berbasis Teks Bahasa Indonesia Dengan Model IndoRoBERTa.")
note1 = st.caption("**Author: Yogie Oktavianus Sihombing**")
note2 = st.write("SENTIMEN adalah sikap, perasaan, atau pandangan ...")
st.info("Masukkan kalimat Anda di kolom bawah dan tekan 'ANALISIS' untuk mulai prediksi. Tekan 'RESET' untuk atur ulang halaman.")

# Klasifikasi sentimen
sentimen = {
  2: 'POSITIF',  
  1: 'NETRAL',
  0: 'NEGATIF'
}

# Klasifikasi emosi
emosi = {
  4: 'SEDIH-KECEWA',
  3: 'SAYANG',
  2: 'SENANG-BAHAGIA',  
  1: 'TAKUT-KHAWATIR',
  0: 'MARAH-JIJIK'
}

# Definisi kategori prediksi
def get_confidence_level(prob):
    if prob >= 0.95:
        return "Kategori **SANGAT TINGGI**, sangat dapat diandalkan"
    elif prob >= 0.85:
        return "Kategori **TINGGI**, umumnya dapat diandalkan"
    elif prob >= 0.70:
        return "Kategori **MODERAT**, cukup dapat diandalkan, tetapi mungkin perlu verifikasi tambahan"
    else:
        return "Kategori **RENDAH**, perlu dipertimbangkan dengan hati-hati"

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
                # Mengabaikan angka dan tanda baca
                if word.isdigit() or re.match(r'^[\W_]+$', word):
                    continue
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

            if warning_count > 7:
                st.error("Warning lebih dari 7, analisis prediksi dihentikan, perbaiki kembali kalimat Anda.")
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

                sentiment_confidence = get_confidence_level(max_sentiment_prob)
                emotion_confidence = get_confidence_level(max_emotion_prob)

                st.write("SENTIMEN:", f"**{sentimen[max_sentiment_index]}**", "--- PREDIKSI:", f"**{max_sentiment_prob:.2%}**", f"({sentiment_confidence}) ---")
                st.write("EMOSI:", f"**{emosi[max_emotion_index]}**", "--- PREDIKSI:", f"**{max_emotion_prob:.2%}**", f"({emotion_confidence}) ---")
        else:
            st.error("Kalimat kurang dari 7 kata, input kembali pada kolom teks.")

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

# Menggabungkan beberapa catatan info
st.markdown("""
<div style="background-color: #002060; padding: 10px; border-radius: 5px; border: 1px solid #003399;">
<h3 style="color: white;">Catatan:</h3>
    <ol style="color: white;">
        <li><strong>Kalimat diharapkan mempunyai konteks, minimal 7 kata.</strong></li>
        <li><strong>Rekomendasi media sosial berbasis teks: Twitter.</strong></li>
        <li><strong>Dimungkinkan analisis dari media sosial lainnya.</strong></li>
        <li><strong>Analisis selain menggunakan bahasa Indonesia tidak disarankan.</strong></li>
        <li><strong>Peringatan yang muncul saat analisis, hanya sebagai reminder.</strong></li>
        <li><strong>Jika muncul lebih dari 7 peringatan, prediksi tidak dijalankan.</strong></li>
    </ol>
</div>
""", unsafe_allow_html=True)
