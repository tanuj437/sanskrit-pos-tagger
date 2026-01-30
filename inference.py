import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def load_model(model_path):
    """
    Load the model and tokenizer from the specified path.
    """
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict(text, tokenizer, model):
    """
    Perform POS tagging on the input text.
    """
    nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    results = nlp(text)
    return results

def format_output(results):
    """
    Format the output for display.
    """
    print(f"\n{'Entity':<20} | {'Label':<10} | {'Score':<10}")
    print("-" * 46)
    for res in results:
        print(f"{res['word']:<20} | {res['entity_group']:<10} | {res['score']:.4f}")

if __name__ == "__main__":
    # Path to your local model or Hugging Face model ID
    # Use "." if running from the directory containing model files
    MODEL_PATH = "." 
    
    # Sample Sanskrit text
    TEST_TEXT = "रामः वनम् गच्छति" # Rama goes to the forest
    
    tokenizer, model = load_model(MODEL_PATH)
    
    if tokenizer and model:
        print(f"\nInput Text: {TEST_TEXT}")
        predictions = predict(TEST_TEXT, tokenizer, model)
        format_output(predictions)
