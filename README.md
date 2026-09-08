# 🔍 Fake Job Posting Detection — NLP & Machine Learning

An NLP and Machine Learning project designed to identify potentially fraudulent job postings by analyzing job description text and classifying postings as **Fake** or **Real**.

## 🎯 Project Overview

Fake job postings can be used for scams, misleading recruitment, and collecting personal information.

This project applies **Natural Language Processing (NLP)** and **Machine Learning** techniques to analyze job-posting text and detect potentially fraudulent listings.

The project also includes an interactive **Streamlit web application** for real-time prediction.

## 🚀 Key Features

- 📝 Accepts job descriptions as input
- 🔤 Converts text into numerical features using **TF-IDF**
- 🤖 Uses **Random Forest** for classification
- 🧠 Provides an additional classification approach using **Gemma3 with Ollama**
- 📊 Displays prediction results through an interactive Streamlit interface
- 📈 Includes model evaluation and visualization

## 📊 Dataset

The project uses a dataset containing job-posting text with corresponding labels.

### Dataset Features

| Feature | Description |
|---|---|
| `text` | Job posting description |
| `label` | Classification label: Fake or Real |

## 🔄 Project Workflow

```text
Job Posting Text
       ↓
Text Preprocessing
       ↓
TF-IDF Vectorization
       ↓
Feature Extraction
       ↓
Random Forest Classification
       ↓
Fake / Real Prediction
       ↓
Streamlit Interface

🧠 Machine Learning Approach
TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) is used to transform job-posting text into numerical features that can be processed by the machine learning model.

Random Forest

A Random Forest classifier is trained using the TF-IDF features to classify job postings as potentially fake or real.

Gemma3 + Ollama

The application also provides an additional text-classification approach using the Gemma3 model through Ollama.

🛠️ Technologies Used
🐍 Python
🐼 Pandas
🔢 NumPy
📊 Matplotlib
📈 Seaborn
🧠 Natural Language Processing (NLP)
🔤 TF-IDF
🌲 Scikit-learn
🤖 Random Forest
🦙 Ollama
🧠 Gemma3
🌐 Streamlit
📈 Model Evaluation

The application provides model evaluation results including:

Accuracy
Classification Report
Confusion Matrix

These metrics are used to evaluate the performance of the classification model.

🌐 Streamlit Application

The project includes an interactive Streamlit application where users can enter a job description and classify it as potentially Fake or Real.

Application Workflow
Enter Job Description
        ↓
Click "Classify"
        ↓
Random Forest Prediction
        +
Gemma3 Prediction
        ↓
Display Results
⚙️ Installation

Clone the repository:
git clone https://github.com/yasiha-analytics/Fake_Job_Posting_Detection.git

Navigate to the project folder:
cd Fake_Job_Posting_Detection

Install the required Python packages:
pip install -r requirements.txt

▶️ Run the Application
Run the Streamlit application using:

streamlit run app.py
Or use the provided batch file:

run.bat
📁 Project Structure
Fake_Job_Posting_Detection/
│
├── app.py
├── fake_job_dataset.csv
├── requirements.txt
├── run.bat
└── README.md

💡Skills Demonstrated
Data preprocessing
Exploratory data analysis
Natural Language Processing
Text feature engineering
TF-IDF vectorization
Machine Learning classification
Model evaluation
Data visualization
Streamlit application development
Python programming


📌 Conclusion
This project demonstrates how NLP and Machine Learning can be applied to classify potentially fraudulent job postings and provide an interactive solution for analyzing job descriptions.

👩‍💻 Author
Yash
📊 Data Analyst | SQL | Power BI | Python | Excel
🔗 GitHub
🔗 LinkedIn
