# Sentiment Analysis with BERT and DistilBERT


This project investigates sentiment analysis of movie reviews using two popular pre-trained transformer models: BERT (Bidirectional Encoder Representation from Transformers) and DistilBERT (a smaller, faster adaptation of BERT).<br>
This project explores sentiment analysis to classify movie reviews from the IMDB dataset into positive or negative categories using Hugging Face Pretrained models.

**Note**: You can run the Entire Application availaible Publically on my HuggingFace Space: https://huggingface.co/spaces/MariamKili/SentimentAnalysisSystem


 ![2024-04-12 18_01_16-Window](https://github.com/Kili66/SentimentAnalysis_BERT_DistilBERT/assets/66678981/d7aeecb2-bac4-4b38-9e8d-fd36138274e1)


## What is Sentiment Analysis
Sentiment analysis is a Natural Language Processing (NLP) task that aims to understand the sentiment expressed in a piece of text. It categorizes the sentiment as positive, negative, or neutral. Sentiment analysis helps businesses and organizations gain insights into customer feedback, product reviews, and social media trends.
### BERT and DistilBERT for Sentiment Analysis
  * BERT (Bidirectional Encoder Representations from Transformers): A powerful pre-trained language model developed by Google AI. BERT excels at various NLP tasks, including sentiment analysis.
  * DistilBERT: A smaller and faster version of BERT, designed for deployment on devices with limited resources. DistilBERT retains good performance while offering improved efficiency.
## About the Dataset
  * The project utilizes a public dataset of movie reviews from Kaggle: 
  * The dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  * This dataset contains contains 50K customer reviews and their corresponding sentiment labels (positive or negative).
## Key Dependencies
- `numpy`: For linear algebra operations
- `pandas`: For data processing and CSV file I/O
- `tensorflow`: For building and training the BERT model
- `nltk`: For text preprocessing
- `transformers`: For utilizing pre-trained BERT models and tokenizers
- `sklearn`: For model evaluation metrics

## Methodology
The project implements sentiment analysis using two pre-trained transformer models:

#### 1. BERT (Bidirectional Encoder Representation from Transformers):
  * A powerful pre-trained model for Natural Language Processing tasks.
  * This project uses the bert-base-uncased model.
#### 2. DistilBERT (Distilled Bidirectional Encoder Representations from Transformers):**
  * A smaller and faster version of BERT with comparable performance.
  * This project uses the distilbert-base-uncased model.
The sentiment analysis process involves the following steps:
### 1. Data Loading and Preprocessing:
  * Load the movie review dataset.
  * Clean and pre-process the text data, including:
      * Converting sentiment labels to numeric values.
      * Removing HTML tags and URLs.
      * Lowercasing text.
      * Removing stop words.
### 2. Data Splitting:
  * Split the data into training and testing sets.
### 3. Tokenization and Encoding:
  * The Bert/Distilbert Tokenizer from transformers is used to convert text data into BERT-compatible tokens (subwords).
  * Padding is applied to ensure all sequences have the same length, and truncation is used for longer reviews to fit within the maximum sequence length allowed by BERT.
  * An attention mask is created to distinguish valid word pieces from padding tokens during training.
  * Encode the tokenized text into numerical representations suitable for the model.
### 4. Model Training:
  * Define and compile the BERT and DistilBERT models for sentiment classification.
  * Configure the model's different hyperparameters (learning rate, batch size,
number of epochs) to find an optimal configuration.
  * TensorFlow Datasets: Training and testing data were converted into TensorFlow datasets for efficient training.
  * Early Stopping: Early stopping was implemented to prevent overfitting by monitoring validation loss.
  * The models were trained on the prepared training dataset in batches.
  * Model Compilation:
      * **Optimizer:**: Adam with Learning rate=2e-5
      * **Loss: SparseCategoricalCrossentropy**: This is typically used when your classes are mutually exclusive and the targets are integers. In the case of binary classification, it expects the labels to be provided as integers (0 or 1), and your final layer should have 2 output neurons with a ‘softmax’ activation function.
      * **Metric:** Accuracy to measure the model accuracy
  * Epochs was kept small(Epochs 2) because of the high training time and lack of resources allocation
  * Train the models on the training data.
### 5. Model Evaluation:
  * Evaluate the trained models on the testing data using metrics like F1_score,accuracy, Precision, Recall using Sklearn Classification report
  * Confusion matrices were created to visualize the model's performance.
    
  1. BERT model
   
   ![bert_cm](https://github.com/Kili66/SentimentAnalysis_HuggingFace_DistlBERT/assets/66678981/f9961862-0ee0-4efd-9d64-5e1313cbd9e6)

  3. DistilBERT model:
     
   ![image-1](https://github.com/Kili66/SentimentAnalysis_HuggingFace_DistlBERT/assets/66678981/af4ddb62-c0fb-4434-aab1-938dfc3f4af7)

### 6. Results
* **Accuracy:** 91% of the reviews in the test set were classified correctly. This is similar to the accuracy achieved by the BERT model (91%).
* **Precision:**
  * Negative Class: 89% of the predicted negative reviews were actually negative (slightly lower than BERT's 93%).
  * Positive Class: 93% of the predicted positive reviews were actually positive (slightly higher than BERT's 90%).
* **Recall:**
  * Negative Class: 93% of the actual negative reviews were correctly classified (slightly higher than BERT's 89%).
  * Positive Class: 89% of the actual positive reviews were correctly classified (slightly lower than BERT's 93%).
* **F1-Score:** The F1-score for both classes is around 0.91, indicating a good balance between precision and recall.
  
   ** BERT RESULT
  
   ![bert_cr](https://github.com/Kili66/SentimentAnalysis_HuggingFace_DistlBERT/assets/66678981/45f9d413-40cd-4eab-8bfb-02470ae41366)

   ** DistilBERT Result
  
   ![distilbert](https://github.com/Kili66/SentimentAnalysis_HuggingFace_DistlBERT/assets/66678981/48993b15-7047-46d7-936c-a8c0a20a1d3a)

  
These results suggest that DistilBERT is a viable alternative to BERT for sentiment analysis, offering comparable performance while potentially being faster and more lightweight due to its smaller size.
### 7. Saving the Model:
  * Save the trained BERT model for future use.
### 8. Pushing the Model and Deploy the App to Hugging Face 

 * Model Hub: https://huggingface.co/MariamKili/my_distilbert_model
 * Deployed App: Run the Application on any device using this link: https://huggingface.co/spaces/MariamKili/SentimentAnalysisSystem

### 9. Implement the Model to a Web Application using Gradio
## Running the Project
1. Clone the repository: ``` git clone https://github.com/Kili66/Sentiment_Analysis_BERT_DistilBert.git  ```
2. Create a Virtual Environment conda: ```conda create -p virtualenv python==3.9 -y```
3. Install dependencies: 
   *  ``` pip install -r requirements.txt  ```
4. Run the data_preprocessing notebook inside src: for data cleaning, preprocessing and Visualization
5. Run the sentiment analysis notebook inside src: to train the models
6. Run the app.py script for sentiment analysis user App

## Challenges Encountered
  ### Memory Constraints and Batch Size Selection
  One of the significant challenges encountered during this project was dealing with memory limitations(OOM), particularly when training the BERT model. BERT is a computationally expensive model, and selecting a large batch size can quickly exhaust available GPU memory, leading to Out-of-Memory (OOM) errors.<br>
  To address this challenge, a careful selection of the batch size was essential. Through experimentation, it was found that a batch size of 6 for BERT training effectively utilized the available GPU memory while providing reasonable training speed.
  ### Dependency Errors

  Another challenge encountered was managing compatibility issues between different libraries, particularly TensorFlow, Keras, and other relevant libraries. Ensuring compatible versions of these libraries is crucial for successful model training and execution.
  ### DistilBERT for Efficiency
  While BERT achieved excellent performance, its computational demands posed challenges. To explore a more memory-efficient alternative, the DistilBERT model was investigated. DistilBERT is a smaller and faster variant of BERT, designed for deployment on devices with limited resources.<br>
  The DistilBERT model allowed for a larger batch size of 12 during training compared to BERT's 6. This increase in batch size can potentially accelerate the training process while maintaining accuracy. However, it's important to evaluate the impact of this larger batch size on model performance through validation metrics.<br>
By carefully considering these challenges and strategically selecting hyperparameters like batch size, we were able to train both BERT and DistilBERT models effectively within the constraints of the available computational resources.

## Conclusion
This project successfully explored sentiment analysis of IMDB movie reviews using pre-trained transformer models, BERT and DistilBERT. We implemented a comprehensive methodology involving data pre-processing, text tokenization, model selection, training, and evaluation. Both models achieved a high overall accuracy of 91% on the test set, demonstrating their effectiveness in classifying reviews as positive or negative.<br>
We also addressed significant challenges, including memory limitations during training with large batch sizes. By carefully selecting a batch size of 6 for BERT and 12 for DistilBERT, we were able to train the models effectively on the available GPU resources. Additionally, we ensured compatibility between different libraries to avoid dependency errors.<br>
### Some Future Explorations
 * Hyperparameter Tuning: Systematically adjusting hyperparameters like learning rate and number of epochs could potentially enhance model performance.
 * Regularization Techniques: Techniques like L1/L2 regularization or dropout can help prevent overfitting.
 * Data Augmentation: Artificially increasing the size and diversity of the training data through techniques like back-translation or synonym replacement can potentially lead to improved model generalization.
 * Exploring Other Models: Investigating the performance of other pre-trained transformer models like RoBERTa or XLNet for sentiment analysis on this dataset could provide valuable insights.
 * Containerize the Application using Docker
 * Deploy The Sentiment Analysis App to the production using a cloud service
