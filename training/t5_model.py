import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
 
 
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
 
 
def load_tokenizer_and_model(model_name, device=None):
    """Load the T5 tokenizer + model and move the model to `device`.
 
    `device` defaults to CUDA if available, else CPU.
    """
    device = device or get_device()
 
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
 
    # Truncates vocab size to match the tokenizer, dropping unused padding rows.
    model.resize_token_embeddings(len(tokenizer))
 
    return tokenizer, model
 