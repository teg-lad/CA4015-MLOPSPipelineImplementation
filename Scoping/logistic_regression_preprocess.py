from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords


def pre_process():
    """
    This function carries out the pre-processing required for creating a proof of concept logistic regression model.
    :return: The pre-processed features and corresponding labels.
    """

    # Download and load the nltk English stop words into memory.
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    # Get the path to the data file, up 2 directories and into resources.
    data_path = Path(__file__).parent.parent / "Resources" / "Data" / "raw.csv"

    # Load the data into a Pandas DataFrame, drop the empty index column and get the text and label out.
    data = pd.read_csv(data_path)
    data.drop(["Unnamed: 0"], axis=1, inplace=True)
    texts, labels = data["text"], data["label"]

    # Create a vector of the counts of important words for each of the texts.
    vectorizer = CountVectorizer(analyzer='word', max_features=1000, max_df=0.8, stop_words=stop_words)
    features = vectorizer.fit_transform(texts).toarray()

    return features, labels
