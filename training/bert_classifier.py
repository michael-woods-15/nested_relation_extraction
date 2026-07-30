import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.utils import get_device, SPECIAL_TOKENS
from data.utils import get_num_labels

def load_tokenizer_and_classifier(model_name, device=None):
    """Load the BERT tokenizer + model and move the model to `device`.
 
    `device` defaults to CUDA if available, else CPU.
    """
    device = device or get_device()
    num_labels = get_num_labels()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels = num_labels
    )

    tokenizer.add_tokens(SPECIAL_TOKENS)
    classifier.resize_token_embeddings(len(tokenizer))

    return tokenizer, classifier