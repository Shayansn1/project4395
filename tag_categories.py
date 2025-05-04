import pandas as pd

# Load job postings
df = pd.read_csv("job_postings.csv")

# Define simple keyword-to-category mapping
category_keywords = {
    "ENGINEERING": ["engineer", "developer", "software", "technician", "architect"],
    "SALES": ["sales", "account executive", "business development"],
    "FINANCE": ["finance", "financial", "accountant", "analyst", "investment"],
    "HEALTHCARE": ["nurse", "medical", "healthcare", "clinical", "hospital", "doctor"],
    "HR": ["hr", "recruiter", "human resources", "talent"],
    "TEACHER": ["teacher", "educator", "instructor", "professor", "tutor"],
    "FITNESS": ["fitness", "trainer", "coach", "yoga", "gym"],
    "PUBLIC-RELATIONS": ["marketing", "pr", "public relations", "communications"],
    "INFORMATION-TECHNOLOGY": ["it", "information technology", "systems", "support"],
    "ADVOCATE": ["attorney", "lawyer", "legal", "advocate", "paralegal"],
    "AGRICULTURE": ["farmer", "agriculture", "agribusiness", "horticulture"],
    "APPAREL": ["fashion", "retail", "apparel", "stylist", "clothing"],
    "ARTS": ["artist", "arts", "design", "graphic", "creative"]
}

# Tag categories based on title keywords
def infer_category(title):
    title = str(title).lower()
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in title:
                return category
    return "OTHER"

df["Category"] = df["title"].apply(infer_category)

# Save updated CSV
df.to_csv("job_postings_tagged.csv", index=False)
print("✅ Categories added. Output: job_postings_tagged.csv")
