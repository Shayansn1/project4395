
import pandas as pd
import random

# Load categorized resume & job data
resumes = pd.read_csv("parsed_resumes.csv")
jobs = pd.read_csv("job_postings_tagged.csv")

pairs = []

# Generate positive and negative pairs
for _, resume in resumes.iterrows():
    resume_cat = resume["Category"]

    # Positive pair: same category
    same_cat_jobs = jobs[jobs["Category"] == resume_cat]
    if not same_cat_jobs.empty:
        job = same_cat_jobs.sample(1).iloc[0]
        pairs.append({
            "Resume Text": resume["Resume Text"],
            "Job Description": job["description"],
            "Label": 1
        })

    # Negative pair: different category
    diff_cat_jobs = jobs[jobs["Category"] != resume_cat]
    if not diff_cat_jobs.empty:
        job = diff_cat_jobs.sample(1).iloc[0]
        pairs.append({
            "Resume Text": resume["Resume Text"],
            "Job Description": job["description"],
            "Label": 0
        })

# Shuffle and split
pairs_df = pd.DataFrame(pairs)
pairs_df = pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(0.8 * len(pairs_df))
train_df = pairs_df[:split_idx]
test_df = pairs_df[split_idx:]

# Save
train_df.to_csv("train_pairs.csv", index=False)
test_df.to_csv("test_pairs.csv", index=False)

print("Done: train_pairs.csv and test_pairs.csv created with category-aware pairing.")
