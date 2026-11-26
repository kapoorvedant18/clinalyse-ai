# Clinalyse AI

AI-augmented blood report analysis and clinical decision support.

Upload a blood-report PDF. Clinalyse AI extracts the lab values, classifies the report into one of 12 broad clinical categories, shows the probabilities, and writes a short summary with possible next steps.

> **Research prototype. Not a medical device.** It is decision support for doctors and laboratory staff and does not diagnose or replace clinical judgement. The model is trained on synthetic data and has not been clinically validated. See [Limitations](#limitations).

## How it works

```
PDF upload -> Gemini 2.5 Flash (extract values as JSON)
           -> feature vector (missing values kept as NaN + presence flags)
           -> LightGBM (probabilities for 12 classes)
           -> Gemini 2.5 Flash (short conclusion + 4-6 possible next steps)
           -> dashboard
```

1. The user opens the web interface and signs in with an email or as a guest.
2. The user uploads a blood-report PDF on the dashboard.
3. The Flask backend sends the PDF to Gemini 2.5 Flash with a strict JSON prompt. Values not present in the report come back as `null`.
4. The backend builds the model input: 75 raw inputs (age, gender and 73 lab values), plus one presence flag per input, 150 columns in total. Missing values stay `NaN`.
5. The LightGBM model returns probabilities for 12 categories. The highest one is the prediction.
6. A second Gemini call receives the predicted class, confidence and values and returns a short conclusion and next steps.
7. The dashboard shows the prediction, confidence, probability bars, summary, next steps and extracted values grouped by lab panel. A warning appears when the top probability is below 20%.

### Categories

Normal, Diabetes/Metabolic, Hematological, Thyroid, Renal, Liver, Lipid/Cardiovascular, Infection/Inflammation, Nutritional, Electrolyte, Hormonal/Endocrine, Clotting.

### Training approach

- One CSV per category plus a Normal class (synthetic / procedurally generated data).
- 70% train, 15% validation, 15% test, stratified.
- LightGBM multiclass objective, balanced sample weights, early stopping on validation loss.
- **Structured Panel Dropout** on the training split: non-defining panels are dropped with probability 0.75 and individual features inside panels with probability 0.30, to imitate incomplete real-world reports.

## Repository layout

```
.
├── app.py                        Flask API (POST /predict)
├── blood_disorder_classifier.py  Training script
├── asdads.py                     Command-line inference helper (JSON in, JSON out)
├── blood_disorder_lgbm.txt       Trained LightGBM model
├── label_encoder.pkl             Class label encoder
├── feature_cols.pkl              Ordered model input columns
├── requirements.txt              Python dependencies
└── frontend/
    ├── index.html                Landing page
    ├── login.html                Email / guest login
    ├── dashboard.html            Upload and results
    ├── app.js
    └── styles.css
```

## Getting started

### Prerequisites

- Python 3.9 or newer
- A Google Gemini API key (https://aistudio.google.com/)
- Internet access, because extraction and summaries call the Gemini API

### 1. Install

```bash
git clone https://github.com/kapoorvedant18/clinalyse-ai.git
cd clinalyse-ai
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Create a file named `.env` in the project root (it is git-ignored):

```
GEMINI_API_KEY=your_api_key_here
```

### 3. Run the backend

```bash
python app.py
```

The API starts on `http://localhost:5000`.

### 4. Run the frontend

The frontend is static HTML, CSS and JavaScript. Serve the `frontend` folder with any static server, for example:

```bash
cd frontend
python -m http.server 5501
```

Then open `http://localhost:5501`. (VS Code Live Server on port 5501 also works.)

The frontend calls the backend at `http://localhost:5000`, set as `BACKEND_URL` at the top of `frontend/app.js`. Change it if your backend runs elsewhere.

## API

### `POST /predict`

Send the PDF as multipart form data under the key `pdf` (or `file`).

```bash
curl -X POST http://localhost:5000/predict -F "pdf=@report.pdf"
```

Response:

```json
{
  "prediction": "Thyroid",
  "confidence": 0.9123,
  "all_probabilities": { "Thyroid": 0.9123, "Normal": 0.0211, "...": 0.0 },
  "extracted_features": { "TSH": 8.2, "Free_T4": 0.7, "...": null },
  "summary": {
    "conclusion": "...",
    "next_steps": ["...", "..."]
  },
  "low_confidence_warning": false
}
```

Errors return JSON with an `error` message and status 400 (no file) or 500 (extraction, feature building or prediction failed).

## Command-line inference

`asdads.py` runs the model on a JSON file of lab values without the web app:

```bash
python asdads.py input.json [output.json]
```

Keys must match the feature names in `feature_cols.pkl` (for example `TSH`, `hemoglobin`, `fasting_glucose`). Missing keys are treated as not tested.

## Training

```bash
python blood_disorder_classifier.py
```

The script reads one CSV per category and writes `blood_disorder_lgbm.txt`, `label_encoder.pkl` and `feature_cols.pkl`.

The training CSVs are **not included** in this repository, and the script currently uses hardcoded local Windows paths (`C:/Users/.../Downloads/...`). Edit the `FILES` dictionary and the save paths before running it.

## Limitations

- **Synthetic training data.** Real-world performance is unknown. There is no prospective clinical validation, regulatory clearance or confirmed-patient benchmark.
- **Reported results are not reproducible from this repo.** The project presentation reports a weighted F1 of about 91% on prototype data, but the training CSVs are not included and the training script prints accuracy only. Treat those figures as demo results, not clinical accuracy.
- **Single-label model.** It cannot represent several coexisting conditions in one report.
- **Depends on the Gemini API.** Extraction and summaries need a valid key and an internet connection. Unusual PDF layouts, units, reference ranges, incomplete panels or conflicting values can produce unreliable results.
- **Generated next steps are model-written text**, not prescriptions.
- **Not production-grade security.** Login and upload history live in the browser's `localStorage`. There is no server-side authentication, database, consent management or audit logging, so do not use real patient data.

## Known issues

Tracked in the project's Jira board.

- Training maps gender as Male=0 / Female=1, but the extraction prompt in `app.py` asks Gemini for 0=female / 1=male, so the two disagree.
- Validation and test splits are not passed through Panel Dropout, so performance on sparse reports is not measured by the training script.
- The landing page text calls the model "XGBoost"; the implemented model is LightGBM.
- Hardcoded Windows paths in `asdads.py` and `blood_disorder_classifier.py`.
- `app.py` runs with `debug=True` and returns raw exception text on errors.
- Malformed Gemini output is not handled robustly.

## Roadmap

Explanations for why an abnormality is flagged, probability calibration, evaluation on ethically obtained anonymised real data, server-side authentication and privacy controls, multi-label prediction, clinician feedback, and longitudinal trends.

## Team

| Name | Enrolment no. | Focus |
| --- | --- | --- |
| Vedant Kapur | S24CSEU0995 | Website: frontend, Flask backend, Gemini integration |
| Dhruv Singh | S24CSEU0813 | ML model: features, LightGBM training, inference |
| Darsh Goswami | S24CSEU0843 | Dataset: synthetic data, feature schema |

Built for CSET455 Agile Software Development, Bennett University (SCSET), under Dr. Achyut Shankar Sinha.

## Disclaimer

Clinalyse AI is not a substitute for professional medical advice, diagnosis or treatment. Always consult a qualified healthcare professional.
