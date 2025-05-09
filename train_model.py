import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# === 1. Clean text ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned)

# === 2. Load and preprocess data ===
df = pd.read_csv("annotated.csv")
df = df.dropna(subset=['tweet_text'])
df['cleaned_text'] = df['tweet_text'].apply(clean_text)

# Focus on top 8 categories
top_categories = df['category'].value_counts().nlargest(8).index
df_top = df[df['category'].isin(top_categories)]

# Encode labels
label_encoder = LabelEncoder()
df_top['category_encoded'] = label_encoder.fit_transform(df_top['category'])

# Split features and labels
X = df_top['cleaned_text']
y = df_top['category_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === 4. Balance classes with SMOTE ===
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

# === 5. Train the model ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train_balanced, y_train_balanced)

# === 6. Save vectorizer, model, and label encoder ===
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Training complete. Model, vectorizer, and label encoder saved.")