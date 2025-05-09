import joblib
import streamlit as st
import pandas as pd
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download resources
nltk.download('punkt')
nltk.download('stopwords')

# Load model components
with open("logistic_model.pkl", "rb") as f:
    model = joblib.load("logistic_model.pkl")
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = joblib.load("label_encoder.pkl")

# Preprocessing
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["Category Predictor", "Dashboard"])

# --- Category Predictor ---
if option == "Category Predictor":
    st.title("Grievance Category Predictor")
    tweet = st.text_area("Enter a complaint or tweet:")
    if st.button("Predict"):
        cleaned = clean_text(tweet)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)
        category = label_encoder.inverse_transform(pred)[0]
        st.success(f"Predicted Category: *{category}*")

# --- Dashboard ---
elif option == "Dashboard":
    st.title("Grievance Dashboard")

    # Load and clean dataset
    df = pd.read_csv("annotated.csv")
    df = df.dropna(subset=['tweet_text', 'category'])
    df['cleaned_text'] = df['tweet_text'].apply(clean_text)

    # Show class distribution
    st.subheader("Top Categories")
    category_counts = df['category'].value_counts().nlargest(10)
    st.bar_chart(category_counts)

    # Show word cloud
    st.subheader("Word Cloud of All Tweets")
    text = " ".join(df['cleaned_text'])
    wc = WordCloud(width=800, height=300, background_color='white').generate(text)
    st.image(wc.to_array())

    # Show org distribution if available
    if 'organization' in df.columns:
        st.subheader("Complaints by Organization")
        org_counts = df['organization'].value_counts()
        st.bar_chart(org_counts)
elif option == "Dashboard":
    st.title("Grievance Dashboard")

    # Load and clean dataset
    df = pd.read_csv("annotated.csv")
    df = df.dropna(subset=['tweet_text', 'category'])
    df['cleaned_text'] = df['tweet_text'].apply(clean_text)

    # Sidebar filters
    st.sidebar.subheader("Filter Options")

    # Organization filter
    org_options = df['organization'].dropna().unique().tolist()
    selected_orgs = st.sidebar.multiselect("Select Organization(s):", org_options, default=org_options)

    # Category filter
    cat_options = df['category'].unique().tolist()
    selected_cats = st.sidebar.multiselect("Select Category(s):", cat_options, default=cat_options)

    # Keyword search
    keyword = st.sidebar.text_input("Search keyword in tweet (optional):")

    # Apply filters
    filtered_df = df[df['organization'].isin(selected_orgs) & df['category'].isin(selected_cats)]
    if keyword:
        filtered_df = filtered_df[filtered_df['cleaned_text'].str.contains(keyword, case=False, na=False)]

    st.subheader(f"Filtered Complaints: {len(filtered_df)}")
    st.dataframe(filtered_df[['tweet_text', 'category', 'organization']].reset_index(drop=True))

    # Category distribution
    st.subheader("Category Distribution")
    st.bar_chart(filtered_df['category'].value_counts())

    # Word cloud
    st.subheader("Word Cloud")
    all_text = " ".join(filtered_df['cleaned_text'])
    wc = WordCloud(width=800, height=300, background_color='white').generate(all_text)
    st.image(wc.to_array())

    # Organization breakdown
    st.subheader("Organization Distribution")
    st.bar_chart(filtered_df['organization'].value_counts())