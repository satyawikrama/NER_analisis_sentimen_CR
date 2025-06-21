# streamlit_text_input_app.py
import streamlit as st
from transformers import pipeline
import re
import pandas as pd
import matplotlib.pyplot as plt

# Inisialisasi pipeline
@st.cache_resource
def load_pipelines():
    ner = pipeline("ner", model="satyawikrama/ner-tokoh-cerita-rakyat_vol2", aggregation_strategy="simple")
    sentiment = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")
    return ner, sentiment

ner_pipeline, sentiment_pipeline = load_pipelines()

# Fungsi preprocessing
def preprocess_ner(text):
    text = re.sub(r'[^\w\s.,;:!?\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def simple_sentence_tokenizer(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# Antarmuka Streamlit
st.title("Analisis Sentimen Tokoh pada Cerita Rakyat")

input_text = st.text_area("Masukkan Teks Cerita Rakyat di sini:", height=300)

if st.button("Analisis"):
    if input_text.strip() == "":
        st.warning("Tolong masukkan teks terlebih dahulu.")
    else:
        cleaned_text = preprocess_ner(input_text)
        sentences = simple_sentence_tokenizer(cleaned_text)

        results = []
        for sentence in sentences:
            try:
                ner_result = ner_pipeline(sentence)
                tokoh = [ent['word'] for ent in ner_result if ent['entity_group'] == 'PER']
                if tokoh:
                    sent_result = sentiment_pipeline(sentence)[0]
                    for t in tokoh:
                        results.append({
                            'tokoh': t,
                            'kalimat': sentence,
                            'sentimen': sent_result['label'],
                            'confidence': round(sent_result['score'], 3)
                        })
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses kalimat: {sentence}")

        if not results:
            st.info("Tidak ditemukan tokoh dalam teks yang dianalisis.")
        else:
            df_result = pd.DataFrame(results)
            st.subheader("Hasil Analisis Sentimen")
            st.dataframe(df_result)

            # Hitung jumlah & confidence rata-rata per tokoh per sentimen
            count_df = df_result.groupby(['tokoh', 'sentimen']).size().reset_index(name='count')
            conf_df = df_result.groupby(['tokoh', 'sentimen'])['confidence'].mean().reset_index(name='avg_confidence')

            # Gabungkan
            merged = pd.merge(count_df, conf_df, on=['tokoh', 'sentimen'])

            # Fungsi untuk menentukan dominan
            def get_dominant_sentiment(group):
                max_count = group['count'].max()
                top = group[group['count'] == max_count]
                if len(top) == 1:
                    return top.iloc[0]['sentimen']
                else:
                    # Jika seri, pilih dengan confidence rata-rata tertinggi
                    top_conf = top.sort_values(by='avg_confidence', ascending=False).iloc[0]
                    return top_conf['sentimen']

            dominant_df = merged.groupby('tokoh').apply(get_dominant_sentiment).reset_index(name='dominant_sentiment')

            # Gabungkan lagi untuk visualisasi
            pivot_df = count_df.pivot(index='tokoh', columns='sentimen', values='count').fillna(0)
            pivot_df = pivot_df.merge(dominant_df, on='tokoh')

            # Tampilkan hasil + visualisasi
            st.subheader("Visualisasi Sentimen per Tokoh")
            for _, row in pivot_df.iterrows():
                tokoh = row['tokoh']
                fig, ax = plt.subplots()
                sentiments = ['positive', 'neutral', 'negative']
                values = [row.get(s, 0) for s in sentiments]
                ax.bar(sentiments, values)
                ax.set_title(f"Tokoh: {tokoh} (Dominan: {row['dominant_sentiment']})")
                ax.set_ylabel("Jumlah Kalimat")
                st.pyplot(fig)
