# LoRA-Adapted-Falcon-7B-for-Instacart-Product-Classification

This project focuses on fine-tuning a language model for predicting product departments based on product names using the Instacart dataset. After setting up the environment and preprocessing the data, a pre-trained language model, TinyPixel/Llama-2-7B-bf16-sharded, is selected and configured. The LoRA model is utilized for fine-tuning, with specific training parameters set using the SFTTrainer from the TRL library. The model is then trained on the dataset, and its performance is evaluated by generating text sequences and comparing the predicted departments with the actual departments in a sample of test data. The results demonstrate the model's ability to generate plausible department predictions based on product names. 

# Steps Involved

# Setup and Environment Preparation:
Installation of necessary libraries using pip andMounting Google Drive to access dataset files.
# Data Preparation:
We have to load the product and department CSV files and Merge the datasets to create a unified dataset.And then Splitting the dataset into training and testing subsets.
# Loading the Model:
We have to choose a pre-trained language model (TinyPixel/Llama-2-7B-bf16-sharded) and Configuring the model, tokenizer, and quantization settings.
# Loading the Trainer:
Configuring the LoRA model and its training parameters.We will use the SFTTrainer from TRL library that gives a wrapper around transformers Trainer to easily fine-tune models on instruction based datasets using PEFT adapters and Training the trainer
# Model Evaluation:
Generating text sequences based on a sample of test data and Comparing the predicted departments with the actual departments to evaluate model performance.
