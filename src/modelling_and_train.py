from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
import evaluate
from data_pipeline import TextClassificationDataPipeline
from datasets import Dataset

class TextClassifier:
    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
        train_dataset: Dataset,
        test_dataset: Dataset,
    ):
        """
        Initializes the TextClassifier.

        Args:
            pretrained_model_name: The name of the pre-trained model to use.
            num_labels: The number of labels in the classification task.
            id2label: A dictionary mapping label IDs to label names.
            label2id: A dictionary mapping label names to label IDs.
            train_dataset: The training dataset.
            test_dataset: The test dataset.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        self.train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        self.test_dataset = test_dataset.map(self.preprocess_function, batched=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.accuracy = evaluate.load("accuracy")

    def preprocess_function(self, examples: dict[str, list]) -> dict[str, list]:
        """Tokenizes the input text."""
        return self.tokenizer(examples["text"], truncation=True, padding=True)

    def compute_metrics(self, eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        """Computes evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.accuracy.compute(predictions=predictions, references=labels)

    def train(
        self,
        output_dir: str,
        learning_rate: float = 2e-5,
        batch_size: int = 32,
        num_train_epochs: int = 1,
        weight_decay: float = 0.01,
    ) -> None:
        """Trains the model."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=100,
            logging_steps=10,
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

# Example usage
if __name__ == "__main__":
    file_path = 'ecommerceDataset.csv'
    columns = ['label', 'text']
    label2id = {"Household": 0, "Books": 1, "Clothing & Accessories": 2, "Electronics": 3}

    pipeline = TextClassificationDataPipeline(file_path, columns, label2id)
    train_dataset, test_dataset = pipeline.load_and_preprocess_data()

    classifier = TextClassifier(
        pretrained_model_name="distilbert-base-uncased",
        num_labels=len(label2id),
        id2label={i: label for i, label in enumerate(label2id)},
        label2id=label2id,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )
    output_dir = './src/saved_models/'
    classifier.train(output_dir)
