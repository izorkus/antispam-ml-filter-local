#!/usr/bin/env python3
import json
import os
from pathlib import Path
import email
from email.parser import BytesParser
from email.policy import default
from email.utils import parseaddr
from bs4 import BeautifulSoup
from io import BytesIO
import re
import argparse

# -------------------------------
# Helper: Extract clean text from email
# -------------------------------
def extract_email_content(file_path):
    try:
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=default).parse(f)

        # Extract sender
        from_raw = msg.get('From', '')
        sender_name, sender_email = parseaddr(from_raw)
        if not sender_name and not sender_email:
            sender_name, sender_email = "(empty)", "(empty)"
        elif not sender_name:
            sender_name = sender_email.split('@')[0].title()  # Fallback

        # Extract subject
        subject = msg.get('Subject', '(no subject)')
        subject = subject.strip() or "(no subject)"

        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
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

                if content_type == "text/html":
                    soup = BeautifulSoup(text, 'html.parser')
                    text = soup.get_text(' ', strip=True)

                body += text + "\n"
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or 'utf-8'
                try:
                    body = payload.decode(charset, errors='replace')
                except:
                    try:
                        body = payload.decode('latin1', errors='replace')
                    except:
                        body = payload.decode('ascii', errors='replace')

                if msg.get_content_type() == "text/html":
                    soup = BeautifulSoup(body, 'html.parser')
                    body = soup.get_text(' ', strip=True)

        # Clean and truncate body
        body = re.sub(r'\s+', ' ', body.strip())
        body_preview = body[:args.max_body_preview] + "..." if len(body) > args.max_body_preview else body

        return {
            "sender_name": sender_name.strip(),
            "sender_email": sender_email.strip(),
            "subject": subject,
            "body_preview": body_preview
        }

    except Exception as e:
        return {
            "sender_name": f"Error: {e}",
            "sender_email": "error",
            "subject": "parsing failed",
            "body_preview": "Unable to parse email content."
        }

# -------------------------------
# Main: Generate Review Report
# -------------------------------
def generate_review_report():
    # Load LLM judgments
    if not os.path.exists(args.llm_labels_json):
        raise FileNotFoundError(f"Labels file not found: {args.llm_labels_json}")

    with open(args.llm_labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)

    print(f"📄 Found {len(labels)} emails to review.\n")

    report = []

    for item in labels:
        file_path = item["file"]
        llm_judgment = item["llm_judgment"].strip().upper()

        print(f"🔍 Processing: {os.path.basename(file_path)} → LLM: {llm_judgment}")

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            content = {
                "sender_name": "(file missing)",
                "sender_email": "(file missing)",
                "subject": "(file missing)",
                "body_preview": "(original file not accessible)"
            }
        else:
            content = extract_email_content(file_path)

        # Add to report
        report.append({
            "filename": os.path.basename(file_path),
            "llm_label": llm_judgment,
            "sender_name": content["sender_name"],
            "sender_email": content["sender_email"],
            "subject": content["subject"],
            "body_preview": content["body_preview"]
        })

    # Save report
    with open(args.output_report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ JSON report generated: {args.output_report_json}")
    print(f"📊 Total entries: {len(report)}")

# -------------------------------
# Optional: Export to CSV (easier for manual review)
# -------------------------------
def export_to_csv():
    import pandas as pd
    df = pd.read_json(args.output_report_json)
    csv_file = args.output_report_json.replace(".json", ".csv")
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\n📎 CSV export saved: {csv_file}")
    print("💡 TIP: Open this CSV in sheets, manual review LLMs labels, and export to CSV again.")
    print("💡 Your dataset will be perfectly labeled! Use this manualy revieved file as a input for train_model.py script.")

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate manual review report from LLM judgments")
    parser.add_argument("--llm-labels-json", default="llm_judge_labels.json", help="Input JSON file containing LLM judgments")
    parser.add_argument("--output-report-json", default="raport_from_labeling.json", help="Output JSON file for the generated report")
    parser.add_argument("--max-body-preview", type=int, default=250, help="Maximum length for email body previews (default: 250)")
    return parser.parse_args()

if __name__ == "__main__":
    global args
    args = parse_arguments()

    generate_review_report()
    export_to_csv()
