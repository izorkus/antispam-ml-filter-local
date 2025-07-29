# Anti-Spam ML Pipeline

This repository contains a complete pipeline for email spam classification using machine learning. The pipeline is designed to be run in a local environment for security, particularly when working with company emails. 

This pipeline is language independent (tested on real polish and english emails dataset). 

Tested on local Minstral-Small-3.2_24B running on llama.cpp and Ollama.

You can deploy model using: https://github.com/izorkus/antispam-ml-filter-microservice repo.

## Key Features

- **Local Environment**: All processing is done locally for security and privacy
- **LLM Integration**: Uses local LLM APIs (like Ollama or LMStudio) for email labeling but it can use any OpenAI compatible API
- **Company-Specific Training**: System prompt includes company summary and market specifics for targeted email classification
- **Fast Predictions**: Optimized for quick model predictions
- **Manual Review**: Manual correction of LLM labels before model training for perfect dataset without contamination

## Pipeline Overview

The pipeline consists of several stages:

1. **Dataset Preparation** (`prepare_dataset.py`)
2. **LLM Labeling** (`llm_labeling.py`)
3. **Manual Review** (`raport_generator.py`)
4. **Model Training** (`train.py`)

### 1. Dataset Preparation

**File**: `prepare_dataset.py`

This script prepares the email dataset for LLM labeling:
- Recursively searches for email files (default: `.eml` files)
- Randomly selects a specified number of emails
- Creates a list of selected files for the next step
- Optionally creates symlinks for easier access

**Usage**:
```bash
python prepare_dataset.py /path/to/email/directory --target_count 1000
```

### 2. LLM Labeling

**File**: `llm_labeling.py`

This script sends emails to a local LLM API for classification:
- Extracts clean text from email files
- Sends email content to LLM API with company-specific system prompt
- Receives and saves LLM judgments (HAM/SPAM/UNCERTAIN)
- Handles API retries and rate limiting

**Configuration**:
- LLM API URL, model name, and API key loaded from `.env` file
- System prompt loaded from `system_prompt.txt`

**Usage**:
```bash
python llm_labeling.py --input_file_list for_labeling.json
```

### 3. Manual Review

**File**: `raport_generator.py`

This script generates a review report for manual correction of LLM labels:
- Extracts email headers and body previews
- Creates a JSON report with LLM judgments
- Exports to CSV for easy manual review
- After manual review, the corrected CSV is used for model training

**Usage**:
```bash
python raport_generator.py --llm-labels-json llm_judge_labels.json
```

### 4. Model Training

**File**: `train.py`

This script trains the spam classification model:
- Loads manually reviewed dataset
- Extracts text features from emails
- Builds and tunes a machine learning pipeline
- Evaluates model performance
- Saves trained model and vectorizer with versioning

**Key Components**:
- TF-IDF vectorizer with stop words
- Logistic Regression classifier with hyperparameter tuning
- Stratified train/validation/test splits
- Performance metrics: accuracy, F1, precision

**Usage**:
```bash
python train.py --input_csv revieved_raport.csv
```

**Pipeline Description**

This is a ***text classification pipeline*** with 2 main steps:

1. **Feature Engineering** (`features` step)
- Type: FeatureUnion (combines multiple feature extraction methods)
- Components: 
  - Text Vectorization: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features
- Configuration:
  - N-grams* 1 to 3 words (captures single words, pairs, and triplets)
  - Vocabulary size: Limited to 50,000 most important terms
  - Text filtering: 
    - Ignores words appearing in less than 2 documents (`min_df=2`)
    - Ignores words appearing in more than 80% of documents (`max_df=0.8`)
    - Removes Polish stop words (common words like "i", "na", "do", etc.)
  - Preprocessing: Converts text to lowercase, removes accents

2. **Classification** (`classifier` step)
- Type: Logistic Regression
- Configuration:
  - Regularization: L2 penalty with strength `C=10.0` (helps prevent overfitting)
  - Class balancing: Automatically adjusts for imbalanced datasets
  - Solver: SAGA algorithm (efficient for large datasets)
  - Random state: Fixed (`321345`) for reproducible results

**Purpose**
This pipeline is designed for **text classification tasks** in Polish language, where it takes raw text input and predicts categorical labels. The TF-IDF vectorizer with n-grams captures both individual important words and meaningful word combinations, while the balanced logistic regression handles classification with good performance on imbalanced datasets.
## Performance

The pipeline has been extensively tested and shows strong performance metrics:

**Dataset Statistics:**
- 270 stop words from stopwords_pl_larg.txt
- 1794 emails loaded
- Final dataset size: 1544 (skiping large files etc.)
- Dataset split: Train (988), Valid (247), Test (309)

**Model Performance:**
- 🧠 Validation Results:
  - F1: 0.9143
  - Precision: 0.9302

- 📈 Test Results:
  - Test Accuracy: 0.9029
  - Test F1: 0.8636
  - Test Precision: 0.8716

**Classification Report (Test):**
```
              precision    recall  f1-score   support

         Ham       0.92      0.93      0.92       198
        Spam       0.87      0.86      0.86       111

    accuracy                           0.90       309
   macro avg       0.90      0.89      0.89       309
weighted avg       0.90      0.90      0.90       309
```

## Security Considerations

- All processing is done locally to protect company email data
- No data is sent to external servers
- LLM API runs locally (Ollama, LMStudio, etc.)
- System prompt includes company-specific context for better classification

## Requirements

- Python 3.8+
- Required libraries: pandas, scikit-learn, numpy, joblib, beautifulsoup4, etc.
- Local LLM API (Ollama, LMStudio, or similar)
- Email dataset in .eml format

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure `.env` file with LLM API credentials

3. Edit `system_prompt.txt` with company-specific instructions in your language
4. Add appropriate stop words file for your language (e.g., `stopwords_pl_larg.txt` for Polish - the system was tested heavily on Polish language)

## Running the Pipeline

1. Prepare dataset:
```bash
python prepare_dataset.py /path/to/emails --target_count 1000
```

2. Label with LLM:
```bash
python llm_labeling.py
```

3. Generate review report:
```bash
python raport_generator.py
```

4. Manually review the CSV file and save corrections

5. Train model:
```bash
python train.py --input_csv revieved_raport.csv
```

## Directory Structure

```
.
├── prepare_dataset.py      # Dataset preparation script
├── llm_labeling.py        # LLM labeling script
├── raport_generator.py    # Manual review report generator
├── train.py               # Model training script
├── .env                   # LLM API credentials
├── system_prompt.txt      # Company-specific system prompt
├── dataset/               # Directory for processed emails
├── model_output/          # Directory for trained models
└── requirements.txt       # Python dependencies
```

## License

This project is licensed under the MIT License.
  
