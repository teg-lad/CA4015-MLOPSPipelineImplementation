## Data

This file contains information on the data stage of our real-world use case.

#### Data Source

In our use case, the data source will be the social media platform that our company operates. We will have access to the raw data and will likely have an abundance of user messages and tweets.

#### Data Cleaning and Labelling

Cleaning for the raw data will depend on the form it takes when extracted. It may be within HTML tags, contain special characters or in some other format. We need to extract the data so it is in a usable format.

Here is an [example]() of what the raw data could have looked like. Processing this would involve using a script like [this](). This gives us clean data that does not contain nulls and has human-readable text. The dataset that we have is already sufficiently clean for use with a deep learning model that extracts features itself, we only need to tokenize the text (more on this later).

Labelling is typically a more manual process and you may initially label samples yourself before scaling up to meet data needs. You can use data labelling services or even create a language model framework to label data using a language models reasoning.

#### Data Exploration

Look in [this notebook]() for some sample data exploration, we aim to understand our data and try to foresee what issues we may run into.

Additionally, having some summary statistics will help us to identify outliers or anomalies when we put our model into production.

#### Data Processing

Data processing is not as difficult when training deep learning model, but it is still needed. For our use case we will need to tokenize the text so that the model can process it. [Here]() is a script showing this.

In classical machine learning, you usually have to spend more time processing the features to extract predictive signal. This can sometimes be the case in deep learning when you make use of multiple features, such as text as well as other metadata.


#### Data Versioning

We will not be carrying out data versioning in this practical as we only have one version of the data, but tools such as [DVC](https://dvc.org/) take a git-like approach to data versioning.