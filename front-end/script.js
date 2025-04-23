async function getMatchScore() {
    const fileInput = document.getElementById('resumeFile');
    const jobDesc = document.getElementById('jobDesc').value;
  
    if (!fileInput.files[0] || !jobDesc.trim()) {
      alert("Please upload a resume and enter a job description.");
      return;
    }
  
    const formData = new FormData();
    formData.append('resume', fileInput.files[0]);
    formData.append('job_description', jobDesc);
  
    const res = await fetch('http://localhost:4000/api/match', {
      method: 'POST',
      body: formData
    });
  
    const data = await res.json();
  
    document.getElementById('score').innerText = `${data.score.toFixed(1)} %`;
    document.getElementById('feedback').innerText = data.feedback;
    document.getElementById('results').style.display = 'block';
  }
  