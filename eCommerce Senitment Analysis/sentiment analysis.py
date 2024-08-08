# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

data = spark.read.format('csv').options(header='true').load('dbfs:/FileStore/flipkart_reviews.csv').toPandas()
print(data)

# COMMAND ----------

print(data.isnull().sum())

# COMMAND ----------

''' Here we used the 
 1. Natural language toolkit (NLTK) - Used in natural language processing. Here we have imported various libraries/functions from it like
    a. nltk.download('stopwords') - used for downloading common stopwords for text preprocessing
    b. stopwords from nltk.corpus -  This corpus contains various common stopwords in different languages , used to filter out stopwords.
    c. stemmer = nltk.SnowballStemmer("english") -  Initializes SnowballStemmer for english lanuage.Stemming reduces words to their base or root form
    d. nltk.download('vader_lexicon') - Used in later cell to import the Vader lexicon model for sentiment analysis
2. re (regular expressions) - Used for pattern matching and text manipulation with regular expressions to search for patterns and perform text replacements.

'''
import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["Review"] = data["Review"].apply(clean)

# COMMAND ----------

# Used plotyexpress library to plot a pie chart of the ratings based on stars given
# 5 being the best and 1 being the worst.

data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')
ratings = data["Rating"].value_counts()
numbers = ratings.index
quantity = ratings.values

import plotly.express as px
figure = px.pie(data, 
             values=quantity, 
             names=numbers,hole = 0.5)
figure.show()

# COMMAND ----------

# This cell shows the most used words in the review.
# Used matplotlib for the wordcloud plot.

text = " ".join(i for i in data.Review)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# COMMAND ----------

# In this cell we have loaded the VADER_lexicon model for sentiment analysis
# We will be analyzing every review in the dataset to predict the sentiment score
# And then we will be storing their scores in "data"
'''
VADER Sentiment Analysis
VADER stands for Valence Aware Dictionary and Sentiment Reasoner. It’s a tool used for sentiment analysis, which is basically a way to figure out if a piece of text is expressing positive, negative, or neutral emotions.
'''''

nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Review"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Review"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Review"]]
data = data[["Review", "Positive", "Negative", "Neutral"]]
print(data.head())

# COMMAND ----------

# Basic function for comparing total positive, negative, neutral sentiments inorder to give an overall sentiment

x = sum(data["Positive"])
y = sum(data["Negative"])
z = sum(data["Neutral"])

def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        print("Positive")
    elif (b>a) and (b>c):
        print("Negative")
    else:
        print("Neutral")

overall_sentiment = sentiment_score(x, y, z)
print(f"Overall_sentiment: {overall_sentiment}")



# COMMAND ----------

print("Positive: ", x)
print("Negative: ", y)
print("Neutral: ", z)

# COMMAND ----------

import plotly.graph_objects as go

# Example sentiment values with labels (replace with actual values)
sentiment_values = [x,z,y]  # Example values
sentiment_labels = ['Positive', 'Neutral', 'Negative']

# Define colors for different sentiment categories
colors = {
    'Positive': 'green',  # Color for positive sentiment
    'Neutral': 'yellow',  # Color for neutral sentiment
    'Negative': 'red'     # Color for negative sentiment
}

# Create a dictionary to map sentiment labels to values
sentiment_dict = dict(zip(sentiment_labels, sentiment_values))

# Determine the ranges and colors for the gauge
sorted_labels = sorted(sentiment_dict.keys(), key=lambda x: sentiment_dict[x])
sorted_values = [sentiment_dict[label] for label in sorted_labels]

min_value = min(sorted_values)
max_value = max(sorted_values) + 100  # Adding some buffer for the gauge

# Determine the overall sentiment label based on the highest sentiment value
overall_sentiment_label = max(sentiment_dict, key=sentiment_dict.get)

# Create the gauge chart
fig = go.Figure()

# Add the gauge chart
fig.add_trace(go.Indicator(
    mode="gauge+number",
    value=max(sentiment_values),
    title={'text': "Overall Sentiment"},
    gauge={
        'axis': {'range': [0, max_value]},
        'steps': [
            {'range': [0, sorted_values[0]], 'color': colors[sorted_labels[0]]},
            {'range': [sorted_values[0], sorted_values[1]], 'color': colors[sorted_labels[1]]},
            {'range': [sorted_values[1], max_value], 'color': colors[sorted_labels[2]]}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': max(sentiment_values)
        }
    }
))

# Display the overall sentiment label as text
fig.add_trace(go.Scatter(
    x=[0], y=[-0.3],
    mode="text",
    text=[f"Overall Sentiment: {overall_sentiment_label}"],
    textposition="bottom center",
    showlegend=False,
    textfont=dict(size=18, color="black")
))

# Update layout to ensure proper spacing and title
fig.update_layout(
    title='Sentiment Analysis',
    height=400,
    width=600,
    showlegend=False
)

# Show the figure
fig.show()

