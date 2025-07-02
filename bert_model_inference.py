"""
simple bert inference script for text classification
"""
from bert_model import load_model, predict_sentiment

# load the saved model and tokenizer
model_path = "./bert_imdb_model"
model, tokenizer = load_model(model_path)


example_texts = [
    "this movie was absolutely fantastic! the acting was superb and the plot was engaging.",
    "worst movie i've ever seen. completely boring and pointless.",
    "the movie was okay, nothing special but not terrible either."
]

for text in example_texts:
    sentiment, confidence = predict_sentiment(text, model, tokenizer)
    print(f"text: {text}")
    print(f"prediction: {sentiment} (confidence: {confidence:.3f})\n")
