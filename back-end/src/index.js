require('dotenv').config();
const express = require('express');
const multer = require('multer');
const axios = require('axios');
const cors = require('cors');
const FormData = require('form-data');
const fs = require('fs');

const app = express();
app.use(cors());
const upload = multer({ dest: 'uploads/' });

app.post('/api/match', upload.single('resume'), async (req, res) => {
  const form = new FormData();
  form.append('resume', fs.createReadStream(req.file.path), req.file.originalname);
  form.append('job_description', req.body.job_description);

  try {
    const response = await axios.post(process.env.NLP_URL, form, {
      headers: form.getHeaders()
    });

    fs.unlinkSync(req.file.path); // remove uploaded file after sending

    res.json(response.data);
  } catch (error) {
    console.error('[Backend error]', error.message);
    res.status(500).json({ error: 'Failed to contact NLP service.' });
  }
});

app.listen(process.env.PORT, () => {
  console.log(`🚀 JS Backend running at http://localhost:${process.env.PORT}`);
});
