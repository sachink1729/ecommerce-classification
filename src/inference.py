from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
from nltk.corpus import stopwords
import string

class TextClassifierInference:
    """
    Class to perform inference using a pre-trained text classifier model.
    """

    def __init__(self, model_path: str):
        """
        Initialize tokenizer and model.

        Args:
            model_path (str): The path to the pre-trained model.
        """
        if not model_path:
            raise ValueError("Model path cannot be empty")

        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def clean_text(self, text: str) -> str:
        """
        Clean the input text by removing punctuation and stopwords.

        Args:
            text (str): The input text.

        Returns:
            str: The cleaned text.
        """
        if not text:
            raise ValueError("Input text cannot be empty")

        text = text.translate(str.maketrans('', '', string.punctuation))
        text = " ".join([word for word in text.split() if word.lower() not in self.stop_words])
        return text

    def predict(self, text: str) -> str:
        """
        Predict the class of the input text.

        Args:
            text (str): The input text.

        Returns:
            str: The predicted label.
        """
        if not text:
            raise ValueError("Input text cannot be empty")

        # Tokenize input text
        inputs = self.tokenizer(self.clean_text(text=text), return_tensors="pt")

        # Perform inference without computing gradients
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Get the predicted class id
        predicted_class_id = logits.argmax().item()

        # Return the predicted label
        return self.model.config.id2label[predicted_class_id]

# Example usage
if __name__ == "__main__":
    model_path = "./src/saved_models"
    inference = TextClassifierInference(model_path)
    text = "micromax phone"
    predicted_label = inference.predict(text)
    print(predicted_label)

