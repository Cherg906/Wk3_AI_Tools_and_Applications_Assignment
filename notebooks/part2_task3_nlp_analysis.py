# part2_task3_nlp_analysis.py
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from spacy import displacy
from wordcloud import WordCloud
from collections import Counter
import random
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load the English language model
print("Loading spaCy model...")
nlp = spacy.load('en_core_web_sm')
print("Model loaded successfully!")

# Sample Amazon product reviews data
print("\nLoading sample data...")
data = {
    'review_id': range(1, 11),
    'review_text': [
        "This product is amazing! The quality exceeded my expectations and it arrived quickly.",
        "Not worth the price. The material feels cheap and broke after a week of use.",
        "Great value for money. I've been using it for a month and it works perfectly.",
        "The product is okay, but the shipping took longer than expected.",
        "Absolutely love it! The design is beautiful and it's very functional.",
        "Terrible experience. The item was damaged upon arrival and customer service was unhelpful.",
        "Good product overall, but the instructions were unclear.",
        "I'm very satisfied with my purchase. It's exactly as described in the pictures.",
        "The quality is decent, but the color is slightly different from what's shown online.",
        "Best purchase I've made this year! Highly recommend to everyone."
    ],
    'rating': [5, 2, 4, 3, 5, 1, 4, 5, 3, 5],
    'category': ['Electronics', 'Home', 'Electronics', 'Home', 'Fashion', 
                'Electronics', 'Home', 'Fashion', 'Fashion', 'Electronics']
}

# Create DataFrame
df = pd.DataFrame(data)
print(f"\nDataset shape: {df.shape}")
print("\nSample data:")
print(df.head())

# Basic Data Exploration
print("\n=== Dataset Information ===")
print(df.info())

print("\n=== Missing Values ===")
print(df.isnull().sum())

# Distribution of ratings
print("\n=== Rating Distribution ===")
plt.figure(figsize=(10, 5))
rating_counts = df['rating'].value_counts().sort_index()
sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='viridis')
plt.title('Distribution of Product Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Category distribution
print("\n=== Category Distribution ===")
plt.figure(figsize=(10, 5))
df['category'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Number of Reviews by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Word cloud of reviews
print("\n=== Word Cloud of Reviews ===")
text = ' '.join(review for review in df['review_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Frequent Words in Reviews')
plt.show()

# Named Entity Recognition (NER)
print("\n=== Named Entity Recognition ===")
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Apply NER to all reviews
df['entities'] = df['review_text'].apply(extract_entities)

# Display entities for the first few reviews
print("\nEntities in first 3 reviews:")
for i, row in df.head(3).iterrows():
    print(f"\nReview {i+1}: {row['review_text']}")
    print("Entities:", row['entities'])

# Sentiment Analysis (Rule-based)
print("\n=== Sentiment Analysis ===")
def analyze_sentiment(text):
    doc = nlp(text)
    # Simple rule-based sentiment analysis
    positive_words = ['amazing', 'great', 'love', 'perfect', 'best', 'good', 'satisfied', 'excellent']
    negative_words = ['terrible', 'bad', 'cheap', 'broke', 'damaged', 'unhelpful']
    
    positive_score = sum(1 for token in doc if token.text.lower() in positive_words)
    negative_score = sum(1 for token in doc if token.text.lower() in negative_words)
    
    if positive_score > negative_score:
        return 'positive'
    elif negative_score > positive_score:
        return 'negative'
    else:
        return 'neutral'

# Apply sentiment analysis
df['sentiment'] = df['review_text'].apply(analyze_sentiment)

# Compare with ratings
print("\n=== Sentiment Analysis Results ===")
print(df[['rating', 'sentiment', 'review_text']].head())

# Visualize sentiment distribution
plt.figure(figsize=(10, 5))
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Save results
df.to_csv('nlp_analysis_results.csv', index=False)
print("\nAnalysis complete! Results saved to 'nlp_analysis_results.csv'")