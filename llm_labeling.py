import os
import re
import json
import time
import requests
import argparse
from pathlib import Path
from email.parser import BytesParser
from email.policy import default
from bs4 import BeautifulSoup
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -------------------------------
# Parse command-line arguments
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='LLM Email Labeling Script')
    parser.add_argument('--input_file_list', default="for_labeling.json",
                        help='Input file list (default: for_labeling.json)')
    parser.add_argument('--output_labels_json', default="llm_judge_labels.json",
                        help='Output labels JSON file (default: llm_judge_labels.json)')
    parser.add_argument('--system_prompt_file', default="system_prompt.txt",
                        help='System prompt file (default: system_prompt.txt)')
    parser.add_argument('--max_email_chars', type=int, default=3000,
                        help='Maximum email characters (default: 3000)')
    parser.add_argument('--max_retries', type=int, default=2,
                        help='Maximum retries for LLM requests (default: 2)')
    parser.add_argument('--retry_delay', type=float, default=1.0,
                        help='Delay between retries (default: 1.0)')
    parser.add_argument('--request_delay', type=float, default=0.5,
                        help='Delay between requests (default: 0.5)')
    return parser.parse_args()

# -------------------------------
# CONFIGURATION (loaded from args)
# -------------------------------
args = parse_args()

INPUT_FILE_LIST = args.input_file_list
OUTPUT_LABELS_JSON = args.output_labels_json
SYSTEM_PROMPT_FILE = args.system_prompt_file
MAX_EMAIL_CHARS = args.max_email_chars
MAX_RETRIES = args.max_retries
RETRY_DELAY = args.retry_delay
REQUEST_DELAY = args.request_delay

# LLM server (loaded from .env)
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# -------------------------------
# Load system prompt
# -------------------------------
if not os.path.exists(SYSTEM_PROMPT_FILE):
    raise FileNotFoundError(f"System prompt file not found: {SYSTEM_PROMPT_FILE}")

with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

# -------------------------------
# Helper: Extract clean text from .eml file
# -------------------------------
def extract_text_from_file(file_path, max_size_kb=500):
    try:
        if os.path.getsize(file_path) > max_size_kb * 1024:
            print(f"⚠️ Skipping large file: {file_path}")
            return ""

        with open(file_path, 'rb') as f:
            content_bytes = f.read()

        msg = BytesParser(policy=default).parse(BytesIO(content_bytes))

        # Extract headers
        header_text = ""
        for hdr in ['From', 'Subject', 'To']:
            if msg[hdr]:
                header_text += f"{hdr}: {msg[hdr]} | "

        # Extract body
        body_text = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get("Content-Disposition"))
                if "attachment" in content_disposition:
                    continue
                payload = part.get_payload(decode=True)
                if not payload:
                    continue
                charset = part.get_content_charset() or 'utf-8'
                try:
                    text = payload.decode(charset, errors='replace')
                except:
                    try:
                        text = payload.decode('latin1', errors='replace')
                    except:
                        text = payload.decode('ascii', errors='replace')
                if part.get_content_type() == "text/html":
                    soup = BeautifulSoup(text, 'html.parser')
                    text = soup.get_text(' ', strip=True)
                body_text += text + " "
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                try:
                    body_text = payload.decode(charset, errors='replace')
                except:
                    try:
                        body_text = payload.decode('latin1', errors='replace')
                    except:
                        body_text = payload.decode('ascii', errors='replace')
                if msg.get_content_type() == "text/html":
                    soup = BeautifulSoup(body_text, 'html.parser')
                    body_text = soup.get_text(' ', strip=True)

        full_text = f"{header_text}\n\n{body_text}".strip()
        return re.sub(r'\s+', ' ', full_text)[:MAX_EMAIL_CHARS]

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return ""

# -------------------------------
# Query LLM via HTTP API
# -------------------------------

def query_llm(email_text, temperature=0.1, max_tokens=8):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    data = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_text}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{LLM_API_URL}/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=45
            )
            if response.status_code == 200:
                resp = response.json()
                content = resp['choices'][0]['message']['content'].strip().upper()
                if "HAM" in content:
                    return "HAM"
                elif "SPAM" in content:
                    return "SPAM"
                else:
                    return "UNCERTAIN"
            else:
                print(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"📡 Request failed (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                return "ERROR"
    return "ERROR"

# -------------------------------
# Main: Load file list and label
# -------------------------------
def main():
    # Load file list
    if not os.path.exists(INPUT_FILE_LIST):
        raise FileNotFoundError(f"Input file list not found: {INPUT_FILE_LIST}")

    with open(INPUT_FILE_LIST, "r", encoding="utf-8") as f:
        file_paths = json.load(f)

    print(f"🧠 Loaded {len(file_paths)} files to label from {INPUT_FILE_LIST}")
    print(f"🚀 Sending to LLM at {LLM_API_URL}")
    print("-" * 60)

    cleaned_labels = []
    total = len(file_paths)

    for idx, file_path in enumerate(file_paths):
        file_path = str(file_path)  # Ensure string
        filename = Path(file_path).name

        print(f"[{idx+1}/{total}] Processing: {filename}")

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            cleaned_labels.append({
                "file": file_path,
                "llm_judgment": "ERROR",
                "error": "File not found"
            })
            continue

        text = extract_text_from_file(file_path)
        if not text:
            cleaned_labels.append({
                "file": file_path,
                "llm_judgment": "ERROR",
                "error": "Empty or failed to extract text"
            })
            continue

        judgment = query_llm(text)
        cleaned_labels.append({
            "file": file_path,
            "llm_judgment": judgment
        })

        if REQUEST_DELAY > 0:
            time.sleep(REQUEST_DELAY)

    # Save results
    with open(OUTPUT_LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(cleaned_labels, f, indent=2, ensure_ascii=False)

    # Log summary
    counts = {}
    for item in cleaned_labels:
        label = item["llm_judgment"]
        counts[label] = counts.get(label, 0) + 1

    print(f"\n✅ LLM labeling complete! Results saved to: {OUTPUT_LABELS_JSON}")
    print("📊 Summary of judgments:")
    for label, count in counts.items():
        print(f"   {label}: {count}")

if __name__ == "__main__":
    main()