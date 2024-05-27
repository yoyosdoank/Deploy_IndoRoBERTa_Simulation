import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter
import os
from langdetect import detect 

def has_vowel(word):
    vowels = 'aeiouAEIOU'
    return any(char in vowels for char in word)

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
user_input = st.text_area('**MASUKKAN KALIMAT DARI MEDIA SOSIAL ATAU LAINNYA:**')
note3 = st.caption("****Harap memasukkan kalimat yang mempunyai konteks, minimal 7 kata dalam 1 kalimat.***")
note4 = st.caption("****Rekomendasi media sosial berbasis teks: Twitter.***")
note5 = st.caption("****Dimungkinkan analisis dari media sosial lainnya.***")
note6 = st.caption("****Analisis selain menggunakan bahasa Indonesia tidak dibenarkan.***")
button = st.button("ANALISIS")
reset_button = st.button("RESET")

    
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

# Jika tombol ditekan, lakukan analisis awal
if user_input and button:
    # Cek apakah input memiliki lebih dari 7 kata
    if len(user_input.split()) > 7:
        # Variabel untuk menghitung jumlah kata dalam bahasa Inggris dan Indonesia
        english_word_count = 0
        indonesian_word_count = 0
        # Cek bahasa setiap kata dalam input
        for word in user_input.split():
            # Cek bahasa dari kata menggunakan library langdetect
            lang = detect(word)
            if lang == 'en':
                english_word_count += 1
            elif lang == 'id':
                indonesian_word_count += 1
            # Inisialisasi variabel untuk melacak karakter dan jumlah kemunculannya dalam kata
            char_count = Counter(word)
            # Periksa apakah ada karakter dengan kemunculan lebih dari 2
            if any(count > 2 for count in char_count.values()):
                st.warning(f"Kata '{word}' memiliki lebih dari 2 huruf yang sama berurutan, dapat mempengaruhi konteks dan prediksi.")
            # Periksa apakah kata tidak memiliki huruf vokal
            if not has_vowel(word):
                st.warning(f"Kata '{word}' tidak memiliki huruf vokal, dapat mempengaruhi konteks dan prediksi.")
        # Cek apakah jumlah kata dalam bahasa Inggris lebih banyak daripada bahasa Indonesia
        if english_word_count > indonesian_word_count:
            st.warning("Kalimat ini dominan dalam bahasa Inggris, dapat mempengaruhi konteks dan prediksi.")
                
        inputs = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')

        # Forward pass through classification layers for model1 and model2
        output1 = model1(**inputs)
        output2 = model2(**inputs)

        logits1 = output1.logits
        logits2 = output2.logits

        # Get the index and probability of the highest predicted sentiment and emotion
        max_sentiment_index = torch.argmax(logits1, dim=1).item()
        max_sentiment_prob = torch.softmax(logits1, dim=1).squeeze()[max_sentiment_index].item()

        max_emotion_index = torch.argmax(logits2, dim=1).item()
        max_emotion_prob = torch.softmax(logits2, dim=1).squeeze()[max_emotion_index].item()

        # Display the highest predicted sentiment and emotion along with their scores
        result_container = st.empty()
        with result_container:
            st.write("Sentimen:", f"**{sentimen[max_sentiment_index]}**", "; Persentase Prediksi:", f"**{max_sentiment_prob:.2%}**")
            st.write("Emosi:", f"**{emosi[max_emotion_index]}**", "; Persentase Prediksi:", f"**{max_emotion_prob:.2%}**")
    else:
        st.error("Panjang 1 kalimat disarankan lebih dari 7 kata untuk memahami konteks dalam kalimat, input kembali pada kolom teks.")

# Jika tombol reset ditekan, hapus input dan hasil sebelumnya
if reset_button:
    user_input = ''
