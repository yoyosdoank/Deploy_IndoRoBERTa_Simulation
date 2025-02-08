import streamlit as st
import numpy as np
from transformers import AutoTokenizer, RobertaTokenizerFast, AutoModelForSequenceClassification, RobertaForSequenceClassification
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

# Fungsi untuk mengecek apakah suatu kata hanya terdiri dari angka atau tanda baca
def is_number_or_punctuation(word):
    return word.isdigit() or re.match(r'^[\W_]+$', word)

# Definisi model Transformers
@st.cache_resource()
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("flax-community/indonesian-roberta-base")
    model1 = AutoModelForSequenceClassification.from_pretrained("yogie27/IndoRoBERTa-Base-Sentiment-Indonesian-Social-Media")
    model2 = AutoModelForSequenceClassification.from_pretrained("yogie27/IndoRoBERTa-Base-Emotion-Indonesian-Social-Media")
    #model3 = AutoModelForSequenceClassification.from_pretrained("yogie27/IndoRoBERTa-Hatespeech-Classifier-Base", token="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")
    return tokenizer, model1, model2
    #model2, model3

tokenizer, model1, model2 = get_model() 

st.image("banner_edit.png", use_container_width=True)
header = st.title("IndoRoBERTa: Klasifikasi Sentimen dan Emosi Berbasis Teks Bahasa Indonesia.")
note1 = st.caption("**Author: Yogie Oktavianus Sihombing**")
note2 = st.write(
    #"EMOSI dan SENTIMEN adalah dua konsep yang berbeda meskipun saling terkait. EMOSI adalah keadaan psikologis yang kompleks dan alami, seperti kebahagiaan atau kemarahan, yang terdiri dari pengalaman subjektif, respons fisiologis, dan respons perilaku. Emosi bersifat mentah dan dapat dipicu oleh berbagai situasi atau kondisi individu. 
    "SENTIMEN adalah sikap mental atau pemikiran yang dipengaruhi oleh emosi, dan lebih terorganisir serta sering kali mencerminkan hubungan dengan objek sosial tertentu, seperti cinta atau kebencian. Sentimen menggabungkan aspek kognitif, fisiologis, dan sosial budaya, menjadikannya lebih dari sekadar respons emosional. (Ivanov, 2023)​")
#note3 = st.write("Ujaran kebencian (HATE SPEECH) adalah tindakan komunikasi dalam bentuk provokasi, hasutan, atau penghinaan terhadap individu atau kelompok berdasarkan aspek seperti ras, warna kulit, etnis, gender, cacat, orientasi seksual, kewarganegaraan, agama, dan lain-lain. Hal ini dapat berupa perkataan, tulisan, atau tindakan yang dilarang karena berpotensi memicu tindakan kekerasan atau prasangka​.(Dictionary.com).")
st.info("Input teks Anda di kolom bawah dan tekan 'ANALISIS' untuk mulai prediksi. Tekan 'RESET' untuk atur ulang halaman.")

# Klasifikasi sentimen
sentimen = {
    2: 'POSITIVE',  
    1: 'NEUTRAL',
    0: 'NEGATIVE'
}

# Klasifikasi emosi
emosi = {
    4: 'SADNESS',
    3: 'LOVE',
    2: 'HAPPY',  
    1: 'FEAR',
    0: 'ANGER'
}

# Klasifikasi hatespeech
hate = {  
    1: 'BUKAN HATE SPEECH',
    0: 'HATE SPEECH'
}

# Definisi kategori prediksi
def get_confidence_level(prob):
    if prob >= 0.90:
        return "**Excellent**"
    elif prob >= 0.80:
        return "**Very High**"  
    elif prob >= 0.70:
        return "**High**"
    elif prob >= 0.60:
        return "**Moderate**"
    else:
        return "**Low**"

# Membuat form
user_input = st.text_area('**MASUKKAN / SALIN TEKS DARI MEDIA SOSIAL:**')
with st.form(key='my_form'):
    button = st.form_submit_button("ANALISIS")
    reset_button = st.form_submit_button("RESET")

    if button and user_input:
        if len(user_input.split()) > 2:
            english_word_count = 0
            indonesian_word_count = 0
            warning_count = 0
            for word in user_input.split():
                # Mengabaikan angka dan tanda baca
                if is_number_or_punctuation(word):
                    continue
                try:
                    lang = detect(word)
                    if lang == 'en':
                        english_word_count += 1
                    elif lang == 'id':
                        indonesian_word_count += 1
                    if has_consecutive_letters(word):
                        st.warning(f"Kata '{word}' memiliki tiga atau lebih huruf yang sama berurutan, mungkin mempengaruhi konteks dan prediksi.")
                        warning_count += 1
                    if not has_vowel(word):
                        st.warning(f"Kata '{word}' tidak memiliki huruf vokal, mungkin mempengaruhi konteks dan prediksi.")
                        warning_count += 1
                except LangDetectException:
                    # Tidak menampilkan peringatan jika kata hanya angka atau tanda baca
                    continue

            if warning_count > 7:
                st.error("Warning lebih dari 7, analisis prediksi dihentikan, perbaiki kembali kalimat Anda.")
            else:
                if english_word_count > indonesian_word_count:
                    st.warning("Kalimat ini dominan dalam bahasa Inggris, mungkin mempengaruhi konteks dan prediksi.")

                inputs = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')

                output1 = model1(**inputs)
                output2 = model2(**inputs)
                #output3 = model3(**inputs)

                logits1 = output1.logits
                logits2 = output2.logits
                #logits3 = output3.logits

                max_sentiment_index = torch.argmax(logits1, dim=1).item()
                max_sentiment_prob = torch.softmax(logits1, dim=1).squeeze()[max_sentiment_index].item()

                max_emotion_index = torch.argmax(logits2, dim=1).item()
                max_emotion_prob = torch.softmax(logits2, dim=1).squeeze()[max_emotion_index].item()

                #max_hatespeech_index = torch.argmax(logits3, dim=1).item()
                #max_hatespeech_prob = torch.softmax(logits3, dim=1).squeeze()[max_hatespeech_index].item()

                sentiment_confidence = get_confidence_level(max_sentiment_prob)
                emotion_confidence = get_confidence_level(max_emotion_prob)
                #hatespeech_confidence = get_confidence_level(max_hatespeech_prob)

                st.write("Sentimen =", f"**{sentimen[max_sentiment_index]}**", ": Score =", f"**{max_sentiment_prob:.2%}**", f"({sentiment_confidence})")
                st.write("Emosi =", f"**{emosi[max_emotion_index]}**", ": Score =", f"**{max_emotion_prob:.2%}**", f"({emotion_confidence})")
                #st.write("Hate Speech =", f"**{hate[max_hatespeech_index]}**", ": Prediksi =", f"**{max_hatespeech_prob:.2%}**", f"({hatespeech_confidence})")
        else:
            st.error("Kalimat kurang dari 3 kata, input kembali pada kolom teks.")

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
        <li><strong>Analisis selain menggunakan bahasa Indonesia tidak disarankan.</strong></li>
        <li><strong>Peringatan berwarna <span style="color: yellow;">KUNING</span> muncul saat analisis sebagai reminder.</strong></li>
            <ul style="margin-left: 25px;">
            <li><strong>Jika <span style="color: yellow;">peringatan</span> muncul < 7, kalimat tetap dapat diprediksi. </strong></li>
            <li><strong>Jika <span style="color: yellow;">peringatan</span> muncul > 7, prediksi tidak berjalan, sesuaikan kembali input teks.</strong></li>
            </ul> 
        <li><strong>Input teks minimal 3 kata.</strong></li>
    </ol>
    
""", unsafe_allow_html=True) 

# margin HTML
st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

st.info("**Disclaimer** : Program ini masih dalam tahap uji coba.")
