import os
import random
import json
from pathlib import Path
import argparse

# -------------------------------
# CONFIGURATION
# -------------------------------
# File extension or pattern (can be set via command line or defaults to .eml)

# Reproducibility (can be set via command line or defaults to random)
SEED = None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Prepare dataset for LLM filtering")
    parser.add_argument("input_dir", help="Root directory containing all .eml files (recursive)")
    parser.add_argument("--output_list", default="for_labeling.json", help="Output file with selected paths (default: for_labeling.json)")
    parser.add_argument("--email_pattern", default=".eml", help="File extension or pattern to search for (default: .eml)")
    parser.add_argument("--target_count", type=int, help="How many emails to select for LLM filtering?")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: random)")
    parser.add_argument("--create_symlinks", action="store_true", help="Create symlinks in a clean folder for easier access (optional)")
    parser.add_argument("--symlink_dir", default="dataset/for_llm_labeling", help="Directory for symlinks (default: dataset/for_llm_labeling)")

    args = parser.parse_args()

    # -------------------------------
    # Find all email files
    # -------------------------------
    input_path = Path(args.input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    # Recursively find files matching pattern
    all_files = []
    for file_path in input_path.rglob(f"*{args.email_pattern}"):
        if file_path.is_file():
            all_files.append(str(file_path.resolve()))

    print(f"📁 Found {len(all_files)} email files in '{args.input_dir}'")

    if len(all_files) <= args.target_count:
        print(f"⚠️  Less than {args.target_count} files — using all.")
        selected_files = all_files
    else:
        # Set seed for reproducibility
        if args.seed is not None:
            random.seed(args.seed)
        else:
            random.seed()
        selected_files = random.sample(all_files, args.target_count)

    print(f"🎯 Selected {len(selected_files)} files for LLM labeling")

    # -------------------------------
    # Save list for next step (LLM filtering)
    # -------------------------------
    with open(args.output_list, "w", encoding="utf-8") as f:
        json.dump(selected_files, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved selection to: {args.output_list}")

    # -------------------------------
    # Optional: Create symlinks (or copies)
    # -------------------------------
    if args.create_symlinks:
        link_dir = Path(args.symlink_dir)
        link_dir.mkdir(parents=True, exist_ok=True)

        print(f"🔗 Creating symlinks in: {link_dir}")
        created = 0
        failed = 0

        for file_path in selected_files:
            src = Path(file_path)
            # Clean filename: remove Maildir suffix like :2,S
            safe_name = src.name.split(":")[0] + ".eml"
            dest = link_dir / safe_name

            if dest.exists():
                dest.unlink()  # Remove old link

            try:
                os.symlink(src.resolve(), dest)
                created += 1
            except OSError as e:
                print(f"❌ Symlink failed for {safe_name}: {e}")
                # Fallback to copy
                try:
                    from shutil import copy2
                    copy2(src, dest)
                    created += 1
                except Exception as ce:
                    print(f"❌ Copy also failed: {ce}")
                    failed += 1

        print(f"✅ Created {created} links/copies | {failed} failed")

    # -------------------------------
    # Summary
    # -------------------------------
    print("\n" + "="*60)
    print("✅ SAMPLING FOR LLM LABELING COMPLETE")
    print("="*60)
    print(f"Total selected: {len(selected_files)}")
    print(f"Target: {args.target_count}")
    print(f"From: {args.input_dir}")
    print(f"Labeling list: {args.output_list}")
    if args.create_symlinks:
        print(f"Access via: {args.symlink_dir}")
    seed_used = args.seed if args.seed is not None else random.getrandbits(32)
    print(f"Seed: {seed_used} (reproducible)")
    print("\n👉 Next: Run your LLM labeling script on the files in the output list")

if __name__ == "__main__":
    main()