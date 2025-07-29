import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score
import re
import email
from email.policy import default
from time import time

def load_stop_words(file_path):
    """Load stop words from a text file (one word per line)"""
    stop_words = set()
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:  # Skip empty lines
                        stop_words.add(word)
            print(f"✅ Loaded {len(stop_words)} stop words from {file_path}")
        except Exception as e:
            raise FileNotFoundError(f"⚠️  Error loading stop words from {file_path}: {e}")

    else:
        raise FileNotFoundError(f"⚠️  Stop words file not found: {file_path}")

    return list(stop_words)

def extract_text_from_file(file_path, max_chars=3000):
    try:
        with open(file_path, 'rb') as f:
            msg = email.message_from_bytes(f.read(), policy=default)

        text = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        try:
                            text += payload.decode(charset, errors='replace')
                        except:
                            try:
                                text += payload.decode('latin1', errors='replace')
                            except:
                                text += payload.decode('ascii', errors='replace')
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                try:
                    text = payload.decode(charset, errors='replace')
                except:
                    try:
                        text = payload.decode('latin1', errors='replace')
                    except:
                        text = payload.decode('ascii', errors='replace')

        text = re.sub(r'\s+', ' ', text.strip())
        return text[:max_chars]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def get_next_version(base_filename):
    """Get the next version number for model files"""
    base_name = Path(base_filename).stem
    ext = Path(base_filename).suffix
    directory = Path(base_filename).parent

    # Find existing versioned files
    pattern = re.compile(rf'{re.escape(base_name)}_(\d+){re.escape(ext)}')
    max_version = -1

    if directory.exists():
        for file in directory.iterdir():
            if file.is_file():
                match = pattern.match(file.name)
                if match:
                    version = int(match.group(1))
                    max_version = max(max_version, version)

    return max_version + 1

def get_versioned_filename(base_filename):
    """Get the next versioned filename"""
    next_version = get_next_version(base_filename)
    base_name = Path(base_filename).stem
    ext = Path(base_filename).suffix
    directory = Path(base_filename).parent
    versioned_name = f"{base_name}_{next_version}{ext}"
    return str(directory / versioned_name)

def main(input_csv="revieved_raport.csv", output_dir="model_output", stop_words_file="polish_stop_words.txt"):

    # Model & vectorizer paths
    MODEL_FILE = os.path.join(output_dir, "spam_model_pipeline.pkl")
    VECTORIZER_FILE = os.path.join(output_dir, "tfidf_vectorizer.pkl")

    # Data split
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2  # Of remaining after test split
    RANDOM_STATE = 321345
    np.random.seed(RANDOM_STATE)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load stop words
    POLISH_STOP_WORDS = load_stop_words(stop_words_file)

    # Load and Prepare Dataset
    print("📊 Loading manually labeled dataset...")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"CSV file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    # Clean column names (in case of extra spaces)
    df.columns = df.columns.str.strip()

    # Convert labels
    df = df[df['llm_label'].isin(['HAM', 'SPAM'])].copy()
    df['label'] = df['llm_label'].map({'HAM': 0, 'SPAM': 1})

    print(f"✅ Loaded {len(df)} emails from CSV")

    # Resolve full file paths (adjust if needed)
    # Assumes CSV has relative or absolute paths in 'filename'
    df['file_path'] = df['filename'].apply(lambda x: x if os.path.isabs(x) else os.path.join("dataset/input", x))

    # Validate files exist
    df['file_exists'] = df['file_path'].apply(os.path.exists)
    missing = df[~df['file_exists']]
    if len(missing) > 0:
        print(f"⚠️  Missing files: {len(missing)}")
        df = df[df['file_exists']].copy()

    # Extract text
    print("📄 Extracting text from emails...")
    df['text'] = df['file_path'].apply(extract_text_from_file)

    # Drop rows with empty text
    df = df[df['text'].str.len() > 10]
    print(f"Final dataset size: {len(df)}")

    # Prepare X and y
    X = df['text'].values
    y = df['label'].values

    # Train / Validation / Test Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"Dataset split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Valid: {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    # Build Pipeline
    print("🧠 Building model pipeline...")
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text', Pipeline([
                ('vectorizer', TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.8,
                    sublinear_tf=True,
                    stop_words=POLISH_STOP_WORDS,
                    norm='l2'
                ))
            ]))
        ])),
        ('classifier', LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000,
            solver='saga'
        ))
    ])

    # Hyperparameter Tuning
    print("⚙️  Tuning hyperparameters...")

    # Create separate parameter distributions for each penalty type
    param_distributions = {
        'features__text__vectorizer__max_features': [20000, 50000],
        'features__text__vectorizer__ngram_range': [(1, 2), (1, 3)],
        'classifier__C': [0.1, 1.0, 10.0]
    }

    # Create separate searches for each penalty type
    best_estimator = None
    best_score = -np.inf

    # Search with l2 penalty (no l1_ratio needed)
    l2_params = param_distributions.copy()
    l2_params['classifier__penalty'] = ['l2']

    l2_search = RandomizedSearchCV(
        pipeline,
        l2_params,
        n_iter=10,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE
    )
    l2_search.fit(X_train, y_train)

    if l2_search.best_score_ > best_score:
        best_score = l2_search.best_score_
        best_estimator = l2_search.best_estimator_

    # Search with elasticnet penalty (requires l1_ratio)
    elasticnet_params = param_distributions.copy()
    elasticnet_params['classifier__penalty'] = ['elasticnet']
    elasticnet_params['classifier__l1_ratio'] = [0.1, 0.15, 0.2]

    elasticnet_search = RandomizedSearchCV(
        pipeline,
        elasticnet_params,
        n_iter=10,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE
    )
    elasticnet_search.fit(X_train, y_train)

    if elasticnet_search.best_score_ > best_score:
        best_score = elasticnet_search.best_score_
        best_estimator = elasticnet_search.best_estimator_

    best_pipeline = best_estimator

    # Evaluate on Validation Set
    y_val_pred = best_pipeline.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)

    print(f"\n🧪 Validation Results:")
    print(f"   F1:       {val_f1:.4f}")
    print(f"   Precision: {val_precision:.4f}")

    # Final Evaluation on Test Set
    print("\n📈 Evaluating on TEST set...")
    y_test_pred = best_pipeline.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)

    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")

    print("\n📋 Classification Report (Test):")
    print(classification_report(y_test, y_test_pred, target_names=['Ham', 'Spam']))

    # Save Model & Vectorizer with Simple Versioning
    # Get versioned filenames for saving
    versioned_model_file = get_versioned_filename(MODEL_FILE)
    versioned_vectorizer_file = get_versioned_filename(VECTORIZER_FILE)

    # Save new model and vectorizer with versioned names
    print(f"\n💾 Saving model to {versioned_model_file}")
    joblib.dump(best_pipeline, versioned_model_file)

    # Save vectorizer separately for reuse
    vectorizer = best_pipeline.named_steps['features'].transformer_list[0][1].named_steps['vectorizer']
    joblib.dump(vectorizer, versioned_vectorizer_file)
    print(f"✅ Vectorizer saved to {versioned_vectorizer_file}")

    print("🎉 Training and evaluation completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train spam classification model")
    parser.add_argument("--input_csv", type=str, default="revieved_raport.csv",
                        help="Path to the input CSV file with labeled data")
    parser.add_argument("--output_dir", type=str, default="model_output",
                        help="Directory to save model outputs")
    parser.add_argument("--stop_words_file", type=str, default="stopwords_eng.txt",
                        help="Path to the stop words file")
    args = parser.parse_args()
    main(args.input_csv, args.output_dir, args.stop_words_file)