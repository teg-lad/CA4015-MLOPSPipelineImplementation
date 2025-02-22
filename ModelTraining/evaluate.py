"""
This script takes all the models in the model directory and evaluates them on the test set to determine which has the
best accuracy.
"""

# Similar imports to the model training script.
from huggingface_hub import login  # This might be needed to log in so you can push a model.
from pathlib import Path
from sklearn.metrics import f1_score
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# Get the path to the root of the repo, so the CA4015... directory
repo_path = Path(__file__).parent.parent

# Add the repo_path to the sys.path.
# This is a list of paths that Python checks when it looks for packages and modules. If we do this then we should be
# able to find and import other scripts from this repo. We will need the dataset script for evalaution so we can be
# consistent with our dataset sampling.
sys.path.append(str(repo_path.absolute()))

# Import the dataset loader, because we want to ensure that the validation set is consistent.
from ModelTraining.train_model import load_dataset


def evaluate():
    """
    This function gets each model from the Models directory, loads it and computes the F1 score. The best model is
    printed along with the score.
    """

    # Get the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the batch size, allows the evaluation to run on smaller GPU and CPU memory.
    batch_size = 64

    # Get the path to the Models directory and the dataset from the current folder.
    model_directory = Path(__file__).parent / "Models"
    dataset_path = Path(__file__).parent.parent / "Resources" / "data.csv"

    # Load the tokenizer, all the models should use the same one.
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

    # Load in the dataset, don't take the train data as we don't need it.
    _, val_texts, _, val_labels = load_dataset(dataset_path)

    # Convert the validation labels to tensor.
    val_labels = torch.tensor(val_labels)

    # Variables to store the best F1 score and the best model.
    best_f1 = None
    best_model = None

    # Iterate over each model, load it in and carry out evalaution.
    for model_path in model_directory.iterdir():

        # Load in the model, we don't need to set the num_labels as the model already has a classification head.
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Move the model to the GPU if available
        model.to(device)

        # List for the current model predictions.
        model_predictions = list()

        # Don't accumulate gradients.
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
                model_predictions.extend(current_predictions)

        # Compute the macro f1 (As we performed sampling this is not too important, but with imbalanced it is important
        # to consider less represented classes.)
        macro_f1 = f1_score(val_labels, model_predictions, average="macro")

        # If we have no best F1 or we have a new best.
        if best_f1 is None or macro_f1 > best_f1:
            # Update the best F1 and the best model.
            best_f1 = macro_f1
            best_model = model_path

    # Output the model name, so we know which model had the best F1 and what the F1 was.
    print(f"The best model in our Models folder is {best_model.name} with an F1 score of {best_f1}")

    # Now that we have the best model, we should push it to HuggingFace so we can download it when we need it.

    # You may need to log in, if you run this you should be able to login
    # login()

    # Alternatively, if you go into setting on HF when logged in, go to access tokens and create a new one with write
    # access. You can pass the hf_token to push_to_hub, and it will use that token to validation access.

    # hf_token = "Your_token_goes_here"

    # Load in the best model
    model = AutoModelForSequenceClassification.from_pretrained(best_model)

    # Push the best model to the Hub under the given name.
    model.push_to_hub("DistilRoBERTaEmotionClassifier")  # Add token=hf_token after filling in your token.

    print("The model has been pushed!")


if __name__ == "__main__":
    evaluate()
