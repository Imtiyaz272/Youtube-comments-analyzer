import joblib
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import googleapiclient.discovery
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import io

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the model architecture
from tensorflow.keras.models import model_from_json

# Load JSON file
with open('sentiment_model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Create model from JSON
# sentiment_model = model_from_json(loaded_model_json)

# # Load weights
# sentiment_model.load_weights('sentiment_model.weights.h5')
# sentiment_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Load trained models and vectorizers
# sentiment_model = load_model('sentiment_model.keras', compile=False)

sentiment_model = load_model("sentiment_model.keras", compile=False)
sentiment_vectorizer = joblib.load("tfidf_vectorizer.pkl")
from keras.optimizers import Adam
sentiment_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

emotion_model = joblib.load("svm_emotion_model.pkl")
emotion_vectorizer = joblib.load("tfidf_vectorizer_em.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Text Preprocessing Function
def clean_text(text):
    if isinstance(text, float):  # Handling NaN or float values
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove special characters
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Extract Video ID from YouTube URL
def extract_video_id(url):
    youtube_regex = r"(?:v=|/)([0-9A-Za-z_-]{11}).*"
    match = re.search(youtube_regex, url)
    return match.group(1) if match else None

# Fetch YouTube Comments
def fetch_youtube_comments(video_url):
    try:
        api_key = "AIzaSyCXH92GttbwaEIsUWTXie5dzYGrPxWR0Aw"
        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
        video_id = extract_video_id(video_url)

        comments = []
        next_page_token = None

        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,  # Maximum allowed per page
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return comments

    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []


def plot_emotion_pie_chart(emotion_percentages):
    labels = list(emotion_percentages.keys())
    sizes = list(emotion_percentages.values())
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=plt.cm.Paired.colors)
    ax.axis("equal")
    st.pyplot(fig)



# Streamlit UI
def main():
    st.title("YouTube Comment Analyzer")
    video_url = st.text_input("Enter YouTube Video URL")

    if st.button("Analyze Comments"):
        if video_url:
            comments = fetch_youtube_comments(video_url)
            if comments:
                df = pd.DataFrame(comments, columns=["text"])
                df["clean_text"] = df["text"].apply(clean_text)

                # **Sentiment Analysis**
                X_sentiment = sentiment_vectorizer.transform(df["clean_text"].astype(str).tolist())
                if X_sentiment.shape[1] != 20621:  # Expected feature count
                    from scipy import sparse
                    # Pad or truncate to match expected shape
                    if X_sentiment.shape[1] < 20621:
                        padding = sparse.csr_matrix((X_sentiment.shape[0], 20621 - X_sentiment.shape[1]))
                        X_sentiment = sparse.hstack([X_sentiment, padding])
                    else:
                        X_sentiment = X_sentiment[:, :20621]
                sentiment_predictions = sentiment_model.predict(X_sentiment)
                df["Sentiment"] = np.argmax(sentiment_predictions, axis=1)

                sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                df["Sentiment_Label"] = df["Sentiment"].map(sentiment_map)

                sentiment_counts = df["Sentiment_Label"].value_counts()
                fig, ax = plt.subplots()
                sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette="pastel")
                ax.set_title("Sentiment Distribution")
                ax.set_ylabel("Number of Comments")
                st.pyplot(fig)

                # **Emotion Detection**
                X_emotion = emotion_vectorizer.transform(df["clean_text"].astype(str).tolist())

                if X_emotion.shape[1] != 12582:
                    if X_emotion.shape[1] < 12582:
                        padding = sparse.csr_matrix((X_emotion.shape[0], 12582 - X_emotion.shape[1]))
                        X_emotion = sparse.hstack([X_emotion, padding])
                    else:
                        X_emotion = X_emotion[:, :12582]

                emotion_predictions = emotion_model.predict(X_emotion)
                df["Emotion"] = emotion_predictions

                # Count frequencies
                emotion_counts = Counter(emotion_predictions)
                print(emotion_counts)
                total_comments = len(df)
                emotion_percentages = {label: (count / total_comments) * 100 for label, count in emotion_counts.items()}

                st.subheader("Emotion Distribution in Pie Chart")
                plot_emotion_pie_chart(emotion_percentages)

                # Display Results
                st.write(df)

            else:
                st.error("No comments found or API limit exceeded.")
        else:
            st.error("Please enter a valid YouTube video link.")

if __name__ == "__main__":
    main()
