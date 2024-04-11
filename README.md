# Sentiment Analysis with BERT and DistilBERT
This project investigates sentiment analysis of movie reviews using two popular pre-trained transformer models: BERT (Bidirectional Encoder Representation from Transformers) and DistilBERT (a smaller, faster adaptation of BERT).

 ![image-2](https://github.com/Kili66/SentimentAnalysis_HuggingFace_DistlBERT/assets/66678981/05c5366b-bbec-4d7d-a9c8-3053a098ecfb)


## What is Sentiment Analysis
Sentiment analysis is a Natural Language Processing (NLP) task that aims to understand the sentiment expressed in a piece of text. It categorizes the sentiment as positive, negative, or neutral. Sentiment analysis helps businesses and organizations gain insights into customer feedback, product reviews, and social media trends.
### BERT and DistilBERT for Sentiment Analysis
  * BERT (Bidirectional Encoder Representations from Transformers): A powerful pre-trained language model developed by Google AI. BERT excels at various NLP tasks, including sentiment analysis.
  * DistilBERT: A smaller and faster version of BERT, designed for deployment on devices with limited resources. DistilBERT retains good performance while offering improved efficiency.
## About the Dataset
  * The project utilizes a public dataset of movie reviews from Kaggle: 
  * The dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  * This dataset contains 50K reviews of movies
## Key Dependencies
- `numpy`: For linear algebra operations
- `pandas`: For data processing and CSV file I/O
- `tensorflow`: For building and training the BERT model
- `nltk`: For text preprocessing
- `transformers`: For utilizing pre-trained BERT models and tokenizers
- `sklearn`: For model evaluation metrics

## Methodology
The project implements sentiment analysis using two pre-trained transformer models:

1. BERT (Bidirectional Encoder Representation from Transformers):
  * A powerful pre-trained model for Natural Language Processing tasks.
  * This project uses the bert-base-uncased model.
2. DistilBERT (Distilled Bidirectional Encoder Representations from Transformers):
  * A smaller and faster version of BERT with comparable performance.
  * This project uses the distilbert-base-uncased model.
The sentiment analysis process involves the following steps:
1. Data Loading and Preprocessing:
  * Load the movie review dataset.
  * Clean and pre-process the text data, including:
      * Converting sentiment labels to numeric values.
      * Removing HTML tags and URLs.
      * Lowercasing text.
      * Removing stop words.
2. Data Splitting:
  * Split the data into training and testing sets.
3. Tokenization and Encoding:
  * Tokenize the text data using the chosen model's tokenizer.
  * Encode the tokenized text into numerical representations suitable for the model.
4. Model Training:
  * Define and compile the BERT and DistilBERT models for sentiment classification.
  * Configure the model's different hyperparameters (learning rate, batch size,
number of epochs) to find an optimal configuration.
  * Model Compilation:
      * **Optimizer:**: Adam with Learning rate=2e-5
      * **Loss: SparseCategoricalCrossentropy**: This is typically used when your classes are mutually exclusive and the targets are integers. In the case of binary classification, it expects the labels to be provided as integers (0 or 1), and your final layer should have 2 output neurons with a ‘softmax’ activation function.
      * **Metric:** Accuracy to measure the model accuracy
  * Epochs was kept small(Epochs 2) because of the high training time and lack of resources allocation
  * Train the models on the training data.
5. Model Evaluation:
  * Evaluate the trained models on the testing data using metrics like F1_score,accuracy, Precision, Recall using Sklearn Classification report
  * Confusion matrix evaluation:
  1. BERT model
    ![bert_cm](https://github.com/Kili66/SentimentAnalysis_HuggingFace_DistlBERT/assets/66678981/f9961862-0ee0-4efd-9d64-5e1313cbd9e6)

  2. DistilBERT model:
    ![alt text](image-1.png)
6. model's predictions
7. Saving the Model:
  * Save the trained BERT model for future use.
8. Pushing the Model to Hugging Face Hub: https://huggingface.co/MariamKili/my_distilbert_model

9. Implement the Model to a Web Application using Gradio
  * Access the public Web App: https://8bf2f8ae82e725538c.gradio.live/
  
## Running the Project
1. Clone the repository: ``` git clone https://github.com/Kili66/Sentiment_Analysis_BERT_DistilBert.git  ```
2. Install dependencies: 
   *  ``` pip install -r requirements.txt  ```
3. Run BERT sentiment analysis:
