from pathlib import Path
from typing import Type, Dict
from datetime import datetime
from itertools import product
from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow


def train_model(dataset_path, hyperparams, device):
    model_to_load = "distilbert/distilroberta-base"
    # model_to_load = "sshleifer/tiny-distilroberta-base"

    # Load in the model with a new classification head and the tokenizer.# sshleifer/tiny-distilroberta-base
    model = AutoModelForSequenceClassification.from_pretrained(model_to_load, num_labels=6)
    tokenizer = AutoTokenizer.from_pretrained(model_to_load)

    model.to(device)

    batch_size, learning_rate = hyperparams["batch_size"], hyperparams["learning_rate"]
    num_epochs = 10

    mlflow_params = {"base_model": model_to_load, "epochs": num_epochs,
                     "batch_size": batch_size, "loss_type": "Negative Log Likelihood", "optimizer": "AdamW",
                     "learning_rate": learning_rate}

    mlflow.log_params(mlflow_params)

    # Load the dataset
    text, labels = load_dataset(dataset_path)

    train_texts, val_texts, train_labels, val_labels = train_test_split(text, labels, random_state=123, stratify=labels)

    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # For each epoch
    for epoch in tqdm(range(10), desc="Epochs"):

        # Track the loss this epoch and the total samples to get the average.
        epoch_loss = 0
        total_samples = 0

        # Training
        for i in tqdm(range(0, len(train_texts), batch_size), desc="Training Steps"):
            batch_texts = train_texts[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

            inputs.to(device)
            batch_labels.to(device)

            outputs = model(**inputs, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss * len(batch_texts)
            total_samples += len(batch_texts)

        epoch_loss = epoch_loss / total_samples

        mlflow.log_metric("training_loss", epoch_loss, step=epoch)

        eval_epoch_loss = 0
        eval_total_samples = 0
        predictions = []

        # Validation
        with torch.no_grad():

            for i in tqdm(range(0, len(train_texts), batch_size), desc="Validation Steps"):

                batch_texts = val_texts[i:i + batch_size]

                val_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                val_outputs = model(**val_inputs)
                predictions.extend(torch.argmax(val_outputs.logits, dim=1).tolist())

                eval_epoch_loss += loss * len(batch_texts)
                eval_total_samples += len(batch_texts)

            val_acc = accuracy_score(val_labels, predictions)

        eval_epoch_loss = eval_epoch_loss / eval_total_samples

        # Log metrics with MLflow
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_loss", eval_epoch_loss, step=epoch)

    # Save the model
    model.save_pretrained("distilroberta_model")


def load_dataset(dataset_path):
    """
    This function loads and returns the dataset to be used.
    :param dataset_path: The path to the dataset we want to load
    :return: The datadset
    """

    df = pd.read_csv(dataset_path)

    return df["text"].tolist(), df["label"].tolist()


if __name__ == "__main__":

    # Define your lists of hyperparameters
    batch_size = [512, 256, 128]
    learning_rate = [2e-5, 2e-4]

    # Generate the Cartesian product of hyperparameters
    hyperparameter_combinations = list(product(batch_size, learning_rate))  # Use `list()` to convert the product
    # iterator to a list

    # Create a dictionary to hold hyperparameter combinations
    hyperparameter_dicts = [{'batch_size': bs, 'learning_rate': lr} for bs, lr, in hyperparameter_combinations]

    # Determine if there is GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load in and transform the dataset
    dataset_path = Path(__file__).parent.parent / "Resources" / "data.csv"

    mlflow.set_tracking_uri("http://localhost:5000/")

    # Set the MLFlow experiment
    mlflow.set_experiment("DistillerRobertaEmotionClassifier")

    # For every set of hyperparameters
    for hyperparams in hyperparameter_dicts:

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_name = "_".join([f"{name}-{value}" for name, value in hyperparams.items()]) + "_" + timestamp
        with mlflow.start_run(run_name=run_name):

            # Train a model
            train_model(dataset_path, hyperparams, device)
        break
