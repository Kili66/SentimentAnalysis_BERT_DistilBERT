import gradio as gr
import tensorflow as tf
from transformers import DistilBertTokenizer
import numpy as np

# Load the saved model and tokenizer
model = tf.keras.models.load_model('path_to_your_saved_model')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

import gradio as gr
# Function to make predictions on new text inputs
def predict_sentiment(text):
    # Tokenize and encode the input text
    encoded_input = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True,
        return_tensors="tf"
    )

    # Make predictions
    output = model(encoded_input)
    probabilities = tf.nn.softmax(output.logits, axis=1).numpy()[0]
    predicted_label = np.argmax(probabilities)
    confidence_score = probabilities[predicted_label]

    # Decode the predicted label
    label = "positive" if predicted_label == 1 else "negative"

    return label, confidence_score

# Create the Gradio interface
text_input = gr.components.Textbox(lines=5, label="Enter your text here")
output_text = gr.components.Textbox(label="Predicted Sentiment")

# Define the Gradio interface
iface=gr.Interface(fn=predict_sentiment, inputs=text_input, outputs=output_text, title="Sentiment Analysis")
# Launch the Gradio app
iface.launch(
    server_port=7860,  # Set the port number for the server
    server_name="0.0.0.0"  # Set the server name (0.0.0.0 allows access from any device in the network)
)