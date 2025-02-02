from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download('stopwords')


class TextClassificationDataPipeline:
    def __init__(
        self,
        file_path: str,
        columns: list[str],
        label_mapping: dict[str, int],
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Initialize the data pipeline.

        Args:
            file_path (str): The path to the CSV file.
            columns (list[str]): The columns in the CSV file.
            label_mapping (dict[str, int]): The mapping of labels to IDs.
            test_size (float, optional): The proportion of data to use for testing. Defaults to 0.2.
            random_state (int, optional): The random state for splitting the data. Defaults to 42.
        """
        self.file_path = file_path
        self.columns = columns
        self.label_mapping = label_mapping
        self.test_size = test_size
        self.random_state = random_state
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        self.train_dataset, self.test_dataset = self.load_and_preprocess_data()

    def load_and_preprocess_data(self) -> tuple[Dataset, Dataset]:
        """Load the dataset, preprocess text, split into train and test sets, and tokenize."""
        dataset = load_dataset('csv', data_files=self.file_path, column_names=self.columns)
        df = dataset['train'].to_pandas()

        # Clean data
        df = df.dropna().drop_duplicates()
        df["label"] = [self.label_mapping[label] for label in df["label"]]

        # Split data
        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)

        # Convert to Dataset format and remove index column
        train_dataset = Dataset.from_pandas(train_df).remove_columns("__index_level_0__")
        test_dataset = Dataset.from_pandas(test_df).remove_columns("__index_level_0__")

        # Print dataset sizes
        print(f"Train size after preprocessing: {len(train_dataset)}, Test size after preprocessing: {len(test_dataset)}")

        # Preprocess datasets
        tokenized_train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_test_dataset = test_dataset.map(self.tokenize_function, batched=True)

        return tokenized_train_dataset, tokenized_test_dataset

    def remove_punctuation_stopwords(self, examples: dict[str, list]) -> dict[str, list]:
        """Remove punctuation and stopwords from the text."""
        def clean_text(text: str) -> str:
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = " ".join([word for word in text.split() if word.lower() not in self.stop_words])
            return text

        examples['text'] = [clean_text(text) for text in examples['text']]
        return examples

    def tokenize_function(self, examples: dict[str, list]) -> dict[str, list]:
        """Tokenize the text using the tokenizer."""
        return self.tokenizer(examples["text"], truncation=True, padding=True)


# Usage
if __name__ == "__main__":
    file_path = './ecommerceDataset.csv'
    columns = ['label', 'text']
    label2id = {"Household": 0, "Books": 1, "Clothing & Accessories": 2, "Electronics": 3}

    pipeline = TextClassificationDataPipeline(file_path, columns, label2id)

