# AI and Graph-Based Detection of Fake Internship and Job Posts

An end-to-end machine learning project that detects suspicious internship and job postings using text signals, metadata checks, and graph-inspired risk analysis.

This repository combines:

- a Flask web app for interactive fake job analysis
- a machine learning pipeline for feature-based fraud detection
- graph-based entity risk scoring using company, location, and email-domain relationships

## Project Overview

Fake internship and job posts often contain repeated fraud patterns such as free email domains, misleading salary claims, payment demands, suspicious links, and reused recruiter identities. This project analyzes those patterns to classify postings as likely real or fake.

The system is built in two practical layers:

- `Web demo`: predicts suspicious jobs from user input using a trained text model plus rule-based checks
- `Research pipeline`: trains a hybrid fraud detector using engineered features and graph-based risk features from structured job data

## Key Features

- Detects suspicious wording like `earn fast`, `instant joining`, and `registration fee`
- Flags unrealistic salary claims and payment-demand language
- Checks recruiter email, phone number, and application link patterns
- Uses graph-style risk scoring for repeated companies, locations, and email domains
- Provides a simple Flask interface for interactive analysis
- Includes sample data and training scripts for reproducible experimentation

## Tech Stack

- `Python`
- `Flask`
- `Pandas`
- `NumPy`
- `scikit-learn`
- `NetworkX`
- `Joblib`
- `HTML`, `CSS`, `Jinja2`

## How It Works

### 1. Data Processing

The training pipeline reads job posting records with fields such as:

- `title`
- `description`
- `company_name`
- `location`
- `salary_text`
- `employment_type`
- `required_experience`
- `email`
- `application_url`

### 2. Feature Engineering

The model extracts useful fraud indicators such as:

- description length
- suspicious keyword frequency
- missing company or location information
- free email domain usage
- email and URL domain mismatch
- fee-related and urgency-related terms

### 3. Graph-Based Risk Scoring

The project also computes relational risk scores using connected entities:

- company risk
- location risk
- email-domain risk

These scores help identify suspicious patterns that may not be obvious from a single post alone.

### 4. Prediction Layer

The final workflow combines:

- machine learning classification
- rule-based fraud checks
- graph-inspired entity analysis

## Repository Structure

```text
.
├── app/
│   ├── app.py
│   └── utils.py
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
├── notebook/
│   └── training.ipynb
├── sample_data/
│   ├── fake_job_postings.csv
│   └── sample_jobs.csv
├── src/
│   └── fake_job_detection/
│       ├── config.py
│       ├── data_utils.py
│       ├── features.py
│       ├── graph_model.py
│       ├── model.py
│       ├── predict.py
│       ├── preprocessing.py
│       └── train.py
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── docs/
│   ├── GITHUB_PROJECT_ASSETS.md
│   └── screenshots/
│       └── README.md
├── requirements.txt
└── README.md
```

## Screenshots

Add your app screenshots inside `docs/screenshots/` using these names:

- `home-page.png`
- `fake-job-result.png`
- `real-job-result.png`
- `project-workflow.png`

Then update this section with image links:

```md
![Home Page](docs/screenshots/home-page.png)
![Fake Job Result](docs/screenshots/fake-job-result.png)
![Real Job Result](docs/screenshots/real-job-result.png)
```

Detailed screenshot guidance is available in [docs/GITHUB_PROJECT_ASSETS.md](/Users/najiyakhatoon/AI_&_Gragh_based_Detection_Of_Fake_InternshipJob/docs/GITHUB_PROJECT_ASSETS.md) and [docs/screenshots/README.md](/Users/najiyakhatoon/AI_&_Gragh_based_Detection_Of_Fake_InternshipJob/docs/screenshots/README.md).

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Flask app

```bash
python app/python3 app.py
```

Open:

```text
http://127.0.0.1:5001
```

### 3. Train the hybrid model

```bash
python -m src.fake_job_detection.train --input sample_data/sample_jobs.csv --output models
```

### 4. Run batch prediction

```bash
python -m src.fake_job_detection.predict --input sample_data/sample_jobs.csv --model-dir models
```

## Sample Use Cases

- Screening internship posts before applying
- Demonstrating fraud detection logic in academic projects
- Building a resume-ready ML + graph-based detection portfolio project
- Extending into a larger fake recruitment intelligence system

## Why This Project Stands Out

- Combines classic ML with graph-based reasoning
- Solves a real-world social impact problem
- Includes both backend training code and a working web interface
- Easy to explain in interviews because the features are interpretable

## Future Improvements

- Add transformer embeddings for richer text understanding
- Use larger real-world datasets for stronger evaluation
- Build network visualizations for suspicious entities
- Replace heuristic graph scores with graph neural networks
- Deploy the app on Render, Hugging Face Spaces, or AWS

## Resume-Friendly Summary

Built an AI and graph-based fake job detection system that analyzes job descriptions, recruiter details, suspicious salary patterns, and entity relationships to classify internship and job postings as real or fraudulent.
