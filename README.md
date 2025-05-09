# Public-Grievance-app
# 🚨 Grievance Category Predictor
A smart, ML-powered Streamlit app that classifies citizen complaints from tweets into relevant *grievance categories* like traffic jam, dirty coaches, no helmet, and more — in seconds.
Built for real-time analysis, this project also features an interactive *dashboard* to explore complaint trends and insights.
---
## ✨ Features
- *Tweet Categorization*: Predict the type of complaint using trained NLP models
- *Interactive Dashboard*: Explore trends by category, organization, and keywords
- *Word Cloud*: Visualize the most frequent complaint terms
- *Multi-filter Support*: Filter by category, organization, or search terms
- *Ready for Deployment*: Easily deployable on Streamlit Cloud
---
## 🚀 Quick Start

1. *Clone the repo*
```bash
git clone https://github.com/your-username/grievance-app.git
cd grievance-app
## run the app locally
you can run this project on your local machine without any deployment

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run grievance_app_project.py
The app will open in your local browser
---
🧠 How It Works
Model: Logistic Regression trained on real-world complaint tweets
Text Processing: Cleaned with NLTK + TF-IDF Vectorization
Category Labels: Automatically encoded and decoded with Scikit-learn
Dashboard: Built with Matplotlib, Seaborn, and WordCloud
---
📂 Project Structure

grievance-app/
├── grievance_app_project.py             # Streamlit app
├── train_model.py           # Model training script
├── annotated.csv            # Sample dataset
├── logistic_model.pkl       # Trained model
├── tfidf_vectorizer.pkl     # Vectorizer
├── label_encoder.pkl        # Label encoder
├── requirements.txt         # Dependencies
└── README.md                # This file

📊 Dashboard Highlights:
Category Distribution: Visual breakdown of complaints
Word Cloud: Top terms used in complaints
Organization Filter: See who’s getting tagged the most
Search Tweets: Filter tweets by keyword or category
---
🧪 Retrain the Model
Want to retrain the model on updated data?
python train_model.py

---
☁ Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Visit streamlit.io/cloud
3. Connect your repo and deploy grievance.py
It’s live in minutes!
---

