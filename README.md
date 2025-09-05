# YouTube Comments Analyzer

A machine learning-powered application for analyzing sentiment and extracting insights from YouTube video comments.

## 📋 Overview

This project provides tools to analyze YouTube comments through sentiment analysis and opinion mining. It combines data collection, preprocessing, machine learning model training, and a user-friendly web interface to deliver comprehensive comment analysis.

## 🚀 Features

- **Sentiment Analysis**: Classify comments as positive, negative, or neutral
- **Opinion Mining**: Extract key opinions and themes from comment data
- **Data Processing**: Clean and preprocess YouTube comment data
- **Model Training**: Train custom machine learning models for comment analysis
- **Web Interface**: Interactive web application for easy analysis


## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Imtiyaz272/youtube-comments-analyzer.git
   cd youtube-comments-analyzer
   ```

2. **Install required dependencies**
   ```bash
   pip install pandas numpy scikit-learn tensorflow flask jupyter matplotlib seaborn nltk textblob
   ```

3. **Download NLTK data** 
   ```python
   import nltk
   nltk.download('vader_lexicon')
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 🎯 Usage

### Running the Web Application

```bash
python app.py
```
