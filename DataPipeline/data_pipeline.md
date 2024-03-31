## Data

This file contains some information on data collection, ingestion, pre-processing and feature engineering.

#### Data Source

The first consideration is where will the data come from? How long will it take to collect? And will it need to be labelled?
These questions are the first you need to answer before you can even begin to plan what data processing you will need.

#### Data Cleaning and Labelling

Once you have collected sufficient data for your first iteration you should begin to clean it into the needed format. This could involve parsing up a JSON file or scraped webpage, removing HTML tags, punctuation and emojis or casting to the correct type. When this is complete you will have data does not contain noise and is split into the correct features.

When the data is cleaned you can begin the labelling process. You can use services such as Amazon Mechanical Turk, use LLMs to label data or manually label data. Ensure that you have labelled the data consistently to avoid conflicting data points.

#### Data Exploration

Now that the data is labelled, you can begin to explore it to better understand the features. This will help you to get an idea of what the average values are for each feature. You can determine if the classes are imbalanced and take this in to account during training.

Having some summary statistics will be useful when it comes time to deploy your model as you can monitor incoming new data and determine if there has been a significant shift in the data distribution.

#### Data Processing

Once you're happy with your data quality and better understand the data it is time to ensure that you have a good predictive signal. Some typical things that are done here may be:

+ Select only features that are correlated with the target variable and not with other features. 
+ Create new features by combining some of the features you already have or augmenting them in some way.
+ Normalise, standardise, one-hot encode, vectorize or tf-idf encode.
+ Augment the data by flipping, rotating or cropping for images.

The aim of data processing is to purify the predictive signal to ensure that the model is able to accurately predict the target label.

#### Data Versioning

As you iterate over the data and modelling stages to improve your model, you will find that versioning your data will allow you to easily determine what data was used to train iterations of the model. This can be useful if you notice degradation or bias being introduced into a model. It can also be useful for auditing purposes as you have a record of what data was used and what that version of the data looked like.

Data provenance is concerned with where your data came from and other metadata relating to the origin or source of the data. Data lineage is concerned with how the data is used, and by whom. Data versioning aims to make both of these clear and avoid having mysterious data that you don't know the origin or usage of.