### NLP Coursework Emotion Classification 

Option 1 was selected for this assignment. I create and compare two models, DistilBERT and a logistic regession model to identify 6emotions off of tweets.

## Hardware notes
Notebooks 01&2: These are lightweight and run quickly on any standard CPU.
Notebook 03:This trains the Transformer. It is computationally heavy. On my laptop (CPU), it took roughly 45-60 minutes to complete the training epochs. If you have a GPU available, it will be much faster.


## File Structure

Code Folder

`requirements.txt`: Run this first to install the necessary libraries (transformers, torch, etc.).
`01_preprocessing.ipynb`:
   - I load the raw data here.
   - Important: I included a manual audit of the labels in this notebook. I found that the brief's mapping was incorrect, so this notebook fixes the labels before saving the clean CSV.
   - It creates two versions of the text: Lemmatized (for the baseline) and Raw (for the transformer)
`02_training_baseline.ipynb`:
    - Runs the TF-IDF vectorization and trains the Logistic Regression model.
    - Achieves approx 85% accuracy.
 `03_training_advanced.ipynb`:
    - Uses the Hugging Face `Trainer` API to fine-tune DistilBERT.
    - I added Early Stopping here to prevent overfitting.
     Achieves approx 92% accuracy.


## Data Folder
`raw_data.csv`: The original file from Blackboard.
`train_cleaned.csv`: The file created by Notebook 01. You need this for the training notebooks to work.

## How to run the code
  Install dependencies:
  pip install -r requirements.txt
  
