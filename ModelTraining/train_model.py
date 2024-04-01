"""
This script is used to train our deep classifier. It uses a Distilled version of the RoBERTa model, which is similar to
the BERT model. It is pre-trained on a Mask Language Modelling (MLM) task to develop good token level embeddings.

We train the model by sampling from our dataset in a stratified way (we avoid class imbalance). We then iterate over the
data, tokenize the text and pass the label to the model during training. We can extract the loss from the output (some
models allow for this, others you need to compute yourself using torch loss functions).

You back-propagate the losses to accumulate the gradients, i.e. what direction a weight needs to be moved to improve the
models performance on the current sample. Then the gradients are scaled by the learning-rate (very small), which when
applied to the weight takes a small step in the direction that increases performance.
"""

# Import the packages that we need.
# datetime - For getting the current time when creating a name to save a model under.
# itertools import product - Needed to expand out our hyperparameter list, so we can iterate over it.
# mlflow - Allows us to track the experiment metrics
# pandas - Needed to read the csv into a DataFrame
# pathlib import Path - Python library for working with paths
# from sklearn.metrics import accuracy_score - Function to take the predictions and ground-truths and compute accuracy.
# from sklearn.model_selection import train_test_split - Split the data into train and validation sets.
# import torch - PyTorch library for general ML functionality.
# from tqdm import tqdm - Library for progress bars
# from transformers import AutoTokenizer, AutoModelForSequenceClassification - HuggingFace transformers for abstracting
# model loading and usage.
from datetime import datetime
from itertools import product
import mlflow
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def train_model(dataset_path, hyperparams, run_name, device):
    """
    This function takes in some parameters for training and trains a RoBERTa model for classifying Twitter/X
    messages/tweets. It has functionality for tracking experiment details on an MLflow instance.

    :param dataset_path: The path to the dataset file
    :param hyperparams: The dictionary of hyerparemeters to train this model.
    :param run_name: The name for this run in MLflow, also becomes the name the model is saved under.
    :param device: The device to train the model on, uses GPU if available, otherwise CPU.
    """

    # The HuggingFace model name, you can see that the user/company name is first, then the model name
    # The Transformers library goes to HuggingFace, finds this model, and downloads it for use.
    model_to_load = "distilbert/distilroberta-base"

    # Load in the model, specifying the number of classes that the classifier should have.
    # This takes the pre-trained model, and puts a new classification head on it with num_labels == outputs logits.
    # i.e. makes the model output 6 values, one for each class in this case.
    model = AutoModelForSequenceClassification.from_pretrained(model_to_load, num_labels=6)

    # Load the tokenizer that was used for training this model, this ensures that when we tokenize words they get
    # converted to the same id as during training. If "the" was 123 during training, the model won't think 321 is "the"
    # now. So be consistent!
    tokenizer = AutoTokenizer.from_pretrained(model_to_load)

    # Move the model to the device, GPU is available.
    # We need the model and inputs to all be on the GPU if we plan to use it for compute.
    model.to(device)

    # Pull out the batch size and learning rate from the hyperparameter dictionary.
    batch_size, learning_rate = hyperparams["batch_size"], hyperparams["learning_rate"]

    # Set the number of epochs and early stopping. If the model does not improve after 5 epochs then stop training.
    num_epochs = 100
    early_stopping = 5

    # Create a dictionary with the parameters, this can then be logged to MLflow and seen on the dashboard.
    mlflow_params = {"base_model": model_to_load, "epochs": num_epochs, "early_stopping": early_stopping,
                     "batch_size": batch_size, "loss_type": "Negative Log Likelihood", "optimizer": "AdamW",
                     "learning_rate": learning_rate}
    mlflow.log_params(mlflow_params)

    # Load the dataset, taking the train and validation texts and labels. This should be deterministic because the
    # random_state is set. This is important as otherwise we can't compare models with differet hyperparameters as there
    # are other factors that introduce change, the data split.
    train_texts, val_texts, train_labels, val_labels = load_dataset(dataset_path)

    # Move the labels to the GPU is available.
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)

    # Create the optimizer, pass all the model parameters as weights to update, and set the learning rate.
    # NOTE: If you wanted to freeze the original layers and just change the classification hear, you could only pass the
    # parameters for that here. The model is a PyTorch model, so I believe you can use model.named_parameters and get
    # the parameters linked to the classification head and pass only them here.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Create a variable to store the best evaluation loss we have seen, this allows us to stop if we have not improved
    # in server epochs.
    best_evaluation_loss = None

    # Create a path to a directory for the Models, create it if it does not exist.
    model_path = Path(__file__).parent / "Models"

    # exist_ok means it won't crash if it does exist, parents means makes directories above it they don't exist either.
    model_path.mkdir(exist_ok=True, parents=True)

    # Iterate over each epoch, we wrap the iterator in a tqdm object to make a progress bar.
    for epoch in tqdm(range(num_epochs), desc="Epochs"):

        # Set the model to train (not always needed, but allows the weights to be edited essentially).
        model.train()

        # Track the loss this epoch and the total samples so we can get the average loss per sample.
        epoch_loss = 0
        total_samples = 0

        # Training loop, iterate over the batches in the training set.
        for i in tqdm(range(0, len(train_texts), batch_size), desc="Training Steps"):

            # Pull out the current batch of text and labels. i.e. the tweet text and the class label.
            batch_texts = train_texts[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            # Pass the text to the tokenizer, using padding, truncation and returning a PyTorch tensor.
            # Padding - If you have 100 text samples of different length, they will all end up being the size of the
            # longest (e.g. 200 tokens) and padding will be added to the end. The model recognizes padding and ignores
            # it, but having a complete matrix is usually better (256, 200) vs list of 256 element of varying length.
            # Truncation -  If a text is longer than the max allowable input of the model, the text is cut off at the
            # limit and only the allowable tokens are returned. As our text is not over the limit we should be fine.
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

            # The inputs have token ids and attention masks. The inputs ids are the id number of a token, so "the" may
            # map to 123. The attention helps the model understand where the input is. So if we had a model that took 2
            # inputs we could create an overlay to show the model where these are.
            # "Dublin City University is a Public entity <SEP> What is DCU?"
            # attn_mask = [1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2]

            # Move the inputs to the GPU if available.
            inputs.to(device)
            batch_labels.to(device)

            # Pass the inputs to the model, and the labels so it can compute the loss.
            outputs = model(**inputs, labels=batch_labels)
            loss = outputs.loss

            # Alternatively, you can get the output.logits and pass these along with the ground truth vector
            # ([1, 0, 0 ,0, 0, 0]) to the torch.nn.functional.nll_loss or torch.nn.functional.cross_entropy function to
            # get the loss.

            # Back propagate the losses
            loss.backward()

            # Take an optimizer step
            optimizer.step()

            # Zero the optimizer so that the gradients are clear for the next training batch.
            optimizer.zero_grad()

            # Keep track of the loss and number of samples for this batch, so we can find the average when we are done.
            # Note: Here we need to move the loss tensor to the CPU, then detach is from the gradients.
            # This is important as otherwise Python thinks we still need the gradients for this batch, and it doesn't
            # clear the memory. If we detach the loss then we can safely use it without worrying about memory leakage.
            # This is when we don't correctly clear the memory after a batch, meaning that over time the memory usage
            # goes up, leading to out of memory issues
            epoch_loss += loss.cpu().detach().numpy() * len(batch_texts)
            total_samples += len(batch_texts)

        # Calculate the average loss per sample.
        epoch_loss = epoch_loss / total_samples

        # Log the epoch loss to MLflow under the name training_loss, the step allows us to plot the loss over time with
        # the step being along the x-axis
        mlflow.log_metric("training_loss", epoch_loss, step=epoch)

        # Create some variables to store the loss and predictions from evaluation.
        eval_epoch_loss = 0
        eval_total_samples = 0
        predictions = []

        # Validation loop, with no grad means that we don't keep track of the forward pass to allow for
        # back-propagation, saving memory.
        with torch.no_grad():

            # Set the model to evaluation, means we cannot update the weights.
            model.eval()

            # Iterate over the validation batches.
            for i in tqdm(range(0, len(val_texts), batch_size), desc="Validation Steps"):

                # Get the current validation batch
                batch_texts = val_texts[i:i + batch_size]
                batch_labels = val_labels[i:i + batch_size]

                # Pass the text into the tokenizer, using the same parameters as before.
                val_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")

                # Move the tokenized text and the labels to the GPU if available.
                val_inputs.to(device)
                batch_labels.to(device)

                # Pass the inputs and labels into the model, getting the outputs.
                val_outputs = model(**val_inputs, labels=batch_labels)

                # Get the max value for each samples output, the samples are along the rows, so we need dim=1 to check
                # them. The max value is the class that was chosen.
                current_predictions = torch.argmax(val_outputs.logits, dim=1).tolist()

                # Extend the prediction list (which started empty) with the list of current predictions.
                predictions.extend(current_predictions)

                # Aggregate the loss and sample count, detaching the loss from the model computation graph.
                eval_epoch_loss += loss.cpu().detach().numpy() * len(batch_texts)
                eval_total_samples += len(batch_texts)

        # Compute the accuracy.
        val_acc = accuracy_score(val_labels, predictions)

        # Compute the average evaluation sample loss.
        eval_epoch_loss = eval_epoch_loss / eval_total_samples

        # Log the validation accuracu and loss to MLflow, using the same epoch. This means we can see the loss for both
        # training and validation, as well as the validation accuracy.
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("val_loss", eval_epoch_loss, step=epoch)

        # If we have recorded an evaluation loss before and the current is better.
        if best_evaluation_loss is None or eval_epoch_loss < best_evaluation_loss:

            # Update the best evaluation loss
            best_evaluation_loss = eval_epoch_loss

            # Reset the count of concurrent epochs without improvement.
            epochs_since_improvement = 0

            # Save the model, so we have the best model when we are done.
            model.save_pretrained((model_path / run_name))

        # If there is no improvement, track that this is an epoch with no improvement.
        else:
            epochs_since_improvement += 1

        # If we have reached the number of allowable epochs without improvement, break the training loop.
        # We have finished training out model!
        if epochs_since_improvement >= early_stopping:
            break


def load_dataset(dataset_path, undersample=True, samples_to_take=None):
    """
    This function loads and returns the dataset at the supplied path. It can undersample classes to avoid class
    imbalance, and can restrict the samples to smaller sizes if there is compute limitations.

    :param dataset_path: The path to the dataset we want to load
    :param undersample: Boolean stating if we should undersample, take as many sample from each class as the smallest
    class has in total.
    :param samples_to_take: The number of samples to take from each class.
    :return: The datadset
    """

    # Read the csv file into a DataFrame.
    df = pd.read_csv(dataset_path)

    # If we want to undersample or we have a number of samples specified.
    if undersample or samples_to_take:

        # If the value of samples to take from each is not specified.
        if not samples_to_take:

            # Calculate the minimum count among all classes, take this many samples from the other classes.
            samples_to_take = df.label.value_counts().min()

        # Define a function to undersample each class. This has to be done here as we need to put samples_to_take in the
        # function, but when we apply it is not easy to pass in the parameter.
        def undersample_class(group):
            # Sample the group, taking the given number of samples.
            return group.sample(samples_to_take, random_state=123)

        # Apply the under sampling function to each class group
        df = df.groupby('label', group_keys=False).apply(undersample_class)

        # Reset the index of the under-sampled DataFrame
        df.reset_index(drop=True, inplace=True)

    # Get the text and labels.
    text, labels = df["text"].tolist(), df["label"].tolist()

    # Split the data into train and test, statifying by label, so train and test should have the same number of each
    # class.
    return train_test_split(text, labels, random_state=123, stratify=labels)


if __name__ == "__main__":
    """This code is run when the script is directly ran."""

    # Define your lists of hyperparameters
    # Feel free to change these, you can even introduce more, but be sure to extract them in
    # the training code and pass them to where they need to be!
    batch_size = [256, 128, 64]
    learning_rate = [1e-6, 1e-5, 1e-4]

    # Generate the Cartesian product of hyperparameters, so 256 with each of [1e-6, 1e-5, 1e-4], then 128... etc.
    hyperparameter_combinations = list(product(batch_size, learning_rate))  # Use `list()` to convert to a list.

    # Create a dictionary to hold hyperparameter combinations
    hyperparameter_dicts = [{'batch_size': bs, 'learning_rate': lr} for bs, lr, in hyperparameter_combinations]

    # Determine if there is GPU available, if not the CPU is used.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the dataset path. Path(__file__) gets the path to the current file, parent goes up one level, so do that twice
    # then we go to the Resources folder and get the data.csv file.
    dataset_path = Path(__file__).parent.parent / "Resources" / "data.csv"

    # Set the MLflow tracking uri, this is where MLflow puts the metrics you track. So make sure MLflow is running!
    # MLflow can be set up on a machine and run as a background process, so you can always use it.
    mlflow.set_tracking_uri("http://localhost:5000/")

    # Set the MLFlow experiment name, this is the tab all the models are metrics are saved under.
    mlflow.set_experiment("DistilRoBERTaEmotionClassifier")

    # Iterate over the parameters, very simple way of doing grid search hyperparameter tuning.
    for hyperparams in hyperparameter_dicts:

        # Get the current datatime, which we can include in the name of this hyperparameter run.
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        run_name = "_".join([f"{name}-{value}" for name, value in hyperparams.items()]) + "_" + timestamp

        # With mlflow, start this run and under this run is where logging happened.
        with mlflow.start_run(run_name=run_name):

            # Train a model with our function.
            train_model(dataset_path, hyperparams, run_name, device)

