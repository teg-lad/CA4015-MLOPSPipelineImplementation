from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import time

from Scoping.logistic_regression_preprocess import pre_process


def train_logistic_regression():
    """
    This function trains a logistic regression model on the dataset. This will act as a proof of concept, which we can
    improve upon with data processing, feature selection and engineering as well as model experimentation.
    """

    # Get the features and labels from our pre-processing script.
    features, labels = pre_process()

    # Split the data into train and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(features, labels, train_size=0.80, random_state=123)

    # Create a LogisticRegression model.
    model = LogisticRegression(max_iter=200)

    # Get the time before we start training.
    start_time = time.time()

    # Fit the model to the data.
    model = model.fit(X=X_train, y=y_train)

    # Get the difference between the current time and the start time to see how long training took.
    training_time = time.time() - start_time
    print(f"The logistic regression model was trained in {training_time} seconds")

    # Test the model on the held-out validation set.
    y_pred = model.predict(X_val)

    # Compute the accuracy of the model.
    model_accuracy = accuracy_score(y_val, y_pred)

    # Print out the accuracy and the confusion matrix.
    print(f"The model achieved an accuracy of {model_accuracy}% on the validation set.\nHere is the confusion matrix:")
    print(confusion_matrix(y_val, y_pred))


if __name__ == "__main__":
    train_logistic_regression()
