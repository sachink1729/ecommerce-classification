# eCommerce-classification 👖👕🤳📱👞

## Overview
This project focuses on classifying e-commerce products using [Distilbert Model](https://huggingface.co/distilbert/distilbert-base-uncased). It leverages Python 3.11, Huggingface's Transformers library, and Pytorch.

## Repository Structure
- **Notebook**: Contains Jupyter Notebook for POC, data analysis, and model training.
- **Scripts**: Python scripts for data preprocessing, model evaluation, training, and inference.
- **Dockerfile**: Configuration to containerize the application.

## Reference and File Access
- [Jupyter Notebook used for POC](https://github.com/sachink1729/ecommerce-classification/blob/main/src/abinbev_assignment_classification_colab.ipynb)
- [Python script for data pipeline](https://github.com/sachink1729/ecommerce-classification/blob/main/src/data_pipeline.py)
- [Python script for inference](https://github.com/sachink1729/ecommerce-classification/blob/main/src/inference.py)
- [Python script for modeling and training](https://github.com/sachink1729/ecommerce-classification/blob/main/src/modelling_and_train.py)
- [Dockerfile](https://github.com/sachink1729/ecommerce-classification/blob/main/Dockerfile)

## Languages Used
- **Jupyter Notebook**: 93.5%
- **Python**: 6.4%
- **Dockerfile**: 0.1%

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sachink1729/ecommerce-classification.git
   ```

2. Install dependencies:
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

## Run
1. Navigate to the repository directory:
   ```bash
   cd ecommerce-classification
   ```
2. Build the Docker container:
   ```bash
   docker build -t ecommerce-classification .
   ```

## Usage
1. Run the Docker container:
   ```bash
   docker run -p 8000:8000 ecommerce-classification
   ```

   OR

2. Run locally
   ```bash
   python ./main.py
   ```

## Evaluation and Run Screenshots
1. Model Evaluation
Trained the Model for 1 epoch with the following parameters:
```bash
{
 learning_rate=2e-5,
 per_device_train_batch_size=32,
 per_device_eval_batch_size=32,
 num_train_epochs=1,
 weight_decay=0.01,
 eval_strategy="steps",
 save_strategy="steps",
 eval_steps=100,
 save_steps=100,
 logging_steps=10
}
```
Accuracy: 95%
![](https://github.com/sachink1729/ecommerce-classification/blob/main/screenshots/test_eval_result.png)

2. Swagger UI
Access at: /swagger
![](https://github.com/sachink1729/ecommerce-classification/blob/main/screenshots/swagger%20api%20ui.png)

3. Run query from Swagger UI
Result:
```json
 {
 "text": "iPhone 16",
 "prediction": "Electronics"
 }
```
![](https://github.com/sachink1729/ecommerce-classification/blob/main/screenshots/run%20query%20from%20swagger.png)


## Contributing
Contribute to this project by sending pull requests.

## License
MIT License
