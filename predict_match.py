import torch
from transformers import BertTokenizer, BertModel
from torch import nn
from twin_model import TwinNetwork  # Make sure this matches your saved architecture

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = TwinNetwork().to(device)
model.load_state_dict(torch.load('twin_model.pth', map_location=device))
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prediction function
def predict_match(resume_text, job_text):
    # Tokenize
    inputs1 = tokenizer(resume_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs2 = tokenizer(job_text, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Move to device
    inputs1 = {k: v.to(device) for k, v in inputs1.items()}
    inputs2 = {k: v.to(device) for k, v in inputs2.items()}

    with torch.no_grad():
        output1, output2 = model(inputs1, inputs2)
        cosine_sim = torch.nn.functional.cosine_similarity(output1, output2)
        similarity_percentage = ((cosine_sim.item() + 1) / 2) * 100  # Scale from [-1,1] to [0,100]

    return round(similarity_percentage, 2)

# Example usage
if __name__ == "__main__":
    resume = "Experienced software engineer with background in Python, machine learning, and data analysis."
    job = "Looking for a Python developer with experience in ML and data science."

    match_score = predict_match(resume, job)
    print(f"Match Score: {match_score}%")
