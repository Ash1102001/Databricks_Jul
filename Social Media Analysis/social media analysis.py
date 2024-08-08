# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor


data = spark.read.format('csv').options(header='true').load('dbfs:/FileStore/Instagram_data.csv').toPandas()
display(data)

# COMMAND ----------

# Checking the dataset for any null values
data.isnull().sum()

# COMMAND ----------

# Dropping the null values if any found
data = data.dropna()

# COMMAND ----------

data.info()

# COMMAND ----------

''' Plot of relationship between impression distribution from home 
As we have taken stat = 'density' , the histogram represent the probability density function (PDF) of the data rather than just the counts.'''


plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.histplot(
    data["From Home"], kde=True,
    stat="density", kde_kws=dict(cut=3),
    alpha=.4, edgecolor=(1, 1, 1, .4),
)
plt.show()

# COMMAND ----------

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.histplot(
    data["From Hashtags"], kde=True,
    stat="density", kde_kws=dict(cut=3),
    alpha=.4, edgecolor=(1, 1, 1, .4),
)
plt.show()

# COMMAND ----------

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.histplot(
    data["From Explore"], kde=True,
    stat="density", kde_kws=dict(cut=3),
    alpha=.4, edgecolor=(1, 1, 1, .4),
)
plt.show()

# COMMAND ----------

''' Needed to convert the object type HOME, Hashtags, Explore and others to int'''


data['From Home'] = pd.to_numeric(data['From Home'], errors='coerce')
data['From Hashtags'] = pd.to_numeric(data['From Hashtags'], errors='coerce')
data['From Explore'] = pd.to_numeric(data['From Explore'], errors='coerce')
data['From Other'] = pd.to_numeric(data['From Other'], errors='coerce')

home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()




labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()

# COMMAND ----------

''' Common words used for captions'''

text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# COMMAND ----------

''' Common hashtags used '''

text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# COMMAND ----------

print(data.dtypes)

# COMMAND ----------

''' need to convert the impressions, likes from object type to int type
to later use it in px.scatter for graph '''

data['Impressions'] = pd.to_numeric(data['Impressions'], errors='coerce')
data['Likes'] = pd.to_numeric(data["Likes"], errors= "coerce")
data['Comments'] = pd.to_numeric(data["Comments"], errors= "coerce")
data['Follows'] = pd.to_numeric(data["Follows"], errors= "coerce")
data['Shares'] = pd.to_numeric(data["Shares"], errors= "coerce")
data['Saves'] = pd.to_numeric(data["Saves"], errors= "coerce")
data['Profile Visits'] = pd.to_numeric(data["Profile Visits"], errors= "coerce")

# COMMAND ----------

'''Relationship Between Likes and Impressions'''

figure = px.scatter(data_frame= data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title = "Relationship Between Likes and Impressions")
figure.show()

# COMMAND ----------

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title = "Relationship Between Comments and Total Impressions")
figure.show()

# COMMAND ----------

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title = "Relationship Between Shares and Total Impressions")
figure.show()

# COMMAND ----------

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols", 
                    title = "Relationship Between Post Saves and Total Impressions")
figure.show()

# COMMAND ----------

''' Here we have calculated correlation between impressions and different parameters. we create a correlation matrix and then extract the values from the correlation matrix and then sort them in descending order'''

correlation = data.corr()
print(correlation["Impressions"].sort_values(ascending=False))

# COMMAND ----------

conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)

# COMMAND ----------

figure = px.scatter(data_frame = data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols", 
                    title = "Relationship Between Profile Visits and Followers Gained")
figure.show()

# COMMAND ----------

# DBTITLE 1,o
''' Here we have created a numpy array (that is suitable for using in ml models)
Now we create a NumPy array of the target variable, which is 'Impressions' in this case
After that we split the data in training and test data for model training and testing
random state 42 ensures that the split is the same every time you run the code, which is important for consistent results in experiments.'''


x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

# COMMAND ----------

''' Load the model and passed the parameters for training and testing '''

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)

# COMMAND ----------

''' Here we have loaded our data for which we want a prediction
all the values in np.array corresponds to the features (like ,share, etc)'''
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)

# COMMAND ----------

'''This cell creates a scatter plot to visualize the relationship between the actual and predicted values of a target variable (in this case, "Impressions"). It helps evaluate the performance of the regression model'''


y_pred = model.predict(xtest)
predictions_df = pd.DataFrame({
    'Actual': ytest,
    'Predicted': y_pred
})

# Creating a scatter plot of actual vs predicted values
fig = px.scatter(predictions_df, x='Actual', y='Predicted',
                 labels={'Actual': 'Actual Impressions', 'Predicted': 'Predicted Impressions'},
                 title='Actual vs. Predicted Impressions',
                 trendline='ols')

fig.show()
