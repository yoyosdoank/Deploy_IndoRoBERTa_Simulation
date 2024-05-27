import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource()
def get_model():
    tokenizer = AutoTokenizer.from_pretrained('flax-community/indonesian-roberta-base')
    model1 = AutoModelForSequenceClassification.from_pretrained("yogie27/IndoRoBERTa-Sentiment-Classifier-for-Twitter", token="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")
    model2 = AutoModelForSequenceClassification.from_pretrained("yogie27/IndoRoBERTa-Emotion-Classifier-for-Twitter", token="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")
    return tokenizer,model1,model2

tokenizer,model1,model2 = get_model()

header = st.header("Klasifikasi Sentimen & Emosi Pada Teks Media Sosial Berbahasa Indonesia Dengan Metode Deep Learning")
note1 = st.caption("Author: Yogie Oktavianus Sihombing")
note2 = st.caption("Sentimen adalah sikap, perasaan, atau pandangan yang lebih stabil dan cenderung bertahan lebih lama terhadap seseorang, situasi, atau fenomena tertentu. Sentimen merupakan cerminan dari emosi yang lebih menetap dan terinternalisasi. Sedangkan emosi adalah respons psikologis yang intens, sering kali singkat, terhadap suatu peristiwa atau situasi. Emosi biasanya bersifat sementara dan bisa berubah dengan cepat. -Ivanov, D. (2023).")
def has_vowel(word):
    vowels = 'aeiouAEIOU'
    return any(char in vowels for char in word)
user_input = st.text_area('Masukkan kalimat dari media sosial yang akan dianalisis:')
note3 = st.caption("*Harap memasukkan kalimat yang mempunyai konteks, minimal 5 kata dalam 1 kalimat.")
note4 = st.caption("*Rekomendasi media sosial: Twitter.")
note5 = st.caption("*Dimungkinkan analisis dari media sosial lainnya.")
note6 = st.caption("*Analisis selain menggunakan bahasa Indonesia tidak dibenarkan.")
button = st.button("Lakukan Analisis")

sentimen = {
  2:'Positif',  
  1:'Netral',
  0:'Negatif'
}

emosi = {
  4:'Sedih / Kecewa',
  3:'Cinta / Sayang',
  2:'Senang / Bahagia',  
  1:'Takut / Khawatir',
  0:'Marah / Jijik'
}

if user_input and button:
    if len(user_input.split()) > 7:
        words_without_vowels = []
        for word in user_input.split():
            if not has_vowel(word):
                words_without_vowels.append(word)
                st.warning(f"Kata '{word}' tidak memiliki huruf vokal.")

        if words_without_vowels:
            confirm = st.checkbox("Beberapa kata tidak memiliki huruf vokal. Apakah Anda ingin melanjutkan dengan input ini?")

            if confirm:
                st.warning("Pengguna telah mengonfirmasi untuk melanjutkan.")
                if st.button("Konfirmasi Analisis"):
                    inputs = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')

                    output1 = model1(**inputs)
                    output2 = model2(**inputs)

                    logits1 = output1.logits
                    logits2 = output2.logits

                    max_sentiment_index = torch.argmax(logits1, dim=1).item()
                    max_sentiment_prob = torch.softmax(logits1, dim=1).squeeze()[max_sentiment_index].item()

                    max_emotion_index = torch.argmax(logits2, dim=1).item()
                    max_emotion_prob = torch.softmax(logits2, dim=1).squeeze()[max_emotion_index].item()

                    st.write("Klasifikasi Sentimen:", f"**{sentimen[max_sentiment_index]}**", "- Tingkat Akurasi:", f"**{max_sentiment_prob:.2%}**")
                    st.write("Klasifikasi Emosi:", f"**{emosi[max_emotion_index]}**", "- Tingkat Akurasi:", f"**{max_emotion_prob:.2%}**")
            else:
                st.info("Silakan ubah input Anda jika perlu.")
        else:
            st.success("Semua kata memiliki huruf vokal. Anda dapat melanjutkan dengan analisis.")
            if st.button("Konfirmasi Analisis"):
                inputs = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')

                output1 = model1(**inputs)
                output2 = model2(**inputs)

                logits1 = output1.logits
                logits2 = output2.logits

                max_sentiment_index = torch.argmax(logits1, dim=1).item()
                max_sentiment_prob = torch.softmax(logits1, dim=1).squeeze()[max_sentiment_index].item()

                max_emotion_index = torch.argmax(logits2, dim=1).item()
                max_emotion_prob = torch.softmax(logits2, dim=1).squeeze()[max_emotion_index].item()

                st.write("Klasifikasi Sentimen:", f"**{sentimen[max_sentiment_index]}**", "- Tingkat Akurasi:", f"**{max_sentiment_prob:.2%}**")
                st.write("Klasifikasi Emosi:", f"**{emosi[max_emotion_index]}**", "- Tingkat Akurasi:", f"**{max_emotion_prob:.2%}**")
    else:
        st.error("Panjang kalimat harus lebih dari 5 kata untuk melakukan analisis konteks dalam kalimat.")

if not user_input:
    st.info("Masukkan teks Anda di atas dan tekan 'Analisis' untuk memulai.")
