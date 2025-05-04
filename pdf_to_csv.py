import fitz  # PyMuPDF
import csv
import os

def extract_resume_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def parse_resume(text):
    lines = text.split('\n')
    data = {
        "Job Titles": [],
        "Skills": [],
        "Education": [],
        "Certifications": [],
        "Resume Text": text.strip()
    }

    for i, line in enumerate(lines):
        l = line.lower()
        if "consultant" in l or "manager" in l or "customer service" in l:
            data["Job Titles"].append(line.strip())

        if "skill highlights" in l or l.startswith("skills"):
            for j in range(i+1, min(i+10, len(lines))):
                if lines[j].strip():
                    data["Skills"].append(lines[j].strip())

        if "education and training" in l:
            for j in range(i+1, min(i+10, len(lines))):
                if lines[j].strip():
                    data["Education"].append(lines[j].strip())

        if "certification" in l or "licenses" in l:
            for j in range(i+1, min(i+10, len(lines))):
                if lines[j].strip():
                    data["Certifications"].append(lines[j].strip())

    return data

def process_folder_recursively(root_folder="data", output_csv="parsed_resumes.csv"):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            "Filename", "Category", "Job Titles", "Skills", 
            "Education", "Certifications", "Resume Text"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for dirpath, _, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.endswith(".pdf"):
                    full_path = os.path.join(dirpath, filename)
                    try:
                        text = extract_resume_text(full_path)
                        resume_data = parse_resume(text)
                        resume_data["Filename"] = filename
                        resume_data["Category"] = os.path.basename(dirpath)
                        writer.writerow({
                            "Filename": resume_data["Filename"],
                            "Category": resume_data["Category"],
                            "Job Titles": '; '.join(resume_data["Job Titles"]),
                            "Skills": '; '.join(resume_data["Skills"]),
                            "Education": '; '.join(resume_data["Education"]),
                            "Certifications": '; '.join(resume_data["Certifications"]),
                            "Resume Text": resume_data["Resume Text"]
                        })
                        print(f" Processed: {filename}")
                    except Exception as e:
                        print(f"❌ Failed to process {filename}: {e}")

# === RUN ===
process_folder_recursively()

print(" All resumes processed into parsed_resumes.csv")
