# 🛡️ Email Spam Detection System

A beautiful, interactive web application for email spam detection using advanced Machine Learning algorithms with real-time analysis and comprehensive data visualizations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25+-brightgreen.svg)

## ✨ Features

- 🤖 **Real-time Email Classification** - Instantly classify emails as spam or legitimate (ham)
- 📊 **Interactive Dashboard** - Beautiful web interface with multiple tabs and visualizations
- 🎯 **High Accuracy** - Achieves 98%+ accuracy on test data
- 📈 **Model Comparison** - Compare 4 different ML algorithms (Naive Bayes, Logistic Regression, SVM, Random Forest)
- 💡 **Confidence Scores** - Get probability scores for predictions
- 📉 **Detailed Analytics** - Comprehensive performance metrics and confusion matrices
- 🎨 **Modern UI** - Gradient designs, animations, and responsive layout
- 🔍 **Text Analysis** - View preprocessing steps and feature extraction

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this project to your local machine**

2. **Install required packages:**

```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn nltk joblib --break-system-packages
```

Or use requirements.txt (if provided):
```bash
pip install -r requirements.txt --break-system-packages
```

3. **Make sure you have these files in your project folder:**
   - `spam.csv` (dataset)
   - `train_and_save_model.py` (model training script)
   - `app.py` (Flask web application)
   - `templates/index.html` (web interface)

### Running the Application

**Step 1: Train the Model (First Time Only)**

```bash
python train_and_save_model.py
```

This will:
- Load and preprocess the dataset
- Train 4 different ML models
- Generate all visualizations
- Save the best model
- Create necessary directories and data files

Expected output:
```
✓ Dataset loaded: 5,572 emails
✓ Text preprocessing complete
✓ Training Naive Bayes...
✓ Training Logistic Regression...
✓ Training SVM...
✓ Training Random Forest...
✓ Best Model: SVM
✓ All visualizations generated
```

**Step 2: Start the Web Application**

```bash
python app.py
```

**Step 3: Open Your Browser**

Navigate to: `http://localhost:5000`

## 📁 Project Structure

```
Email-Spam-Detection/
│
├── spam.csv                          # Dataset (5,572 emails)
├── train_and_save_model.py          # Model training script
├── app.py                           # Flask web application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── templates/
│   └── index.html                   # Web interface
│
├── models/                          # Generated models (after training)
│   ├── best_model.pkl              # Best performing model
│   ├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
│   └── stemmer.pkl                 # Porter Stemmer
│
├── data/                           # Generated data (after training)
│   ├── dataset_stats.json          # Dataset statistics
│   ├── text_stats.json             # Text analysis stats
│   ├── model_results.json          # Model comparison results
│   └── classification_report.json  # Detailed metrics
│
└── static/
    └── images/                     # Generated visualizations (after training)
        ├── dataset_distribution.png
        ├── text_analysis.png
        ├── model_comparison.png
        ├── confusion_matrices.png
        └── best_model_metrics.png
```

## 🎨 Web Interface Sections

### 1. **Predict Tab** 🤖
- Enter or paste email text
- Click "Analyze Email" to get instant prediction
- View confidence scores and probabilities
- Try example emails with one click

### 2. **Dataset Tab** 📊
- View dataset statistics (total emails, ham/spam counts)
- See distribution charts
- Analyze text length and word count patterns

### 3. **Models Tab** 🧠
- Compare all 4 ML models
- View confusion matrices
- See best model performance metrics

### 4. **Analysis Tab** 📈
- Interactive charts (using Chart.js)
- Detailed classification metrics
- Training vs testing accuracy comparison

### 5. **About Tab** ℹ️
- Project overview
- Technology stack
- ML pipeline explanation

## 🧪 Example Usage

### Testing Emails

**Spam Example:**
```
Congratulations! You've won a $1000 gift card. Click here to claim now!
```
Expected: ⚠️ SPAM (95%+ confidence)

**Ham Example:**
```
Hey, are we still meeting for lunch tomorrow at noon?
```
Expected: ✅ HAM (95%+ confidence)

### API Endpoints

The application provides REST API endpoints:

**Predict Single Email:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "FREE! Win an iPhone now!"}'
```

**Get Dataset Stats:**
```bash
curl http://localhost:5000/api/dataset-stats
```

**Get Model Results:**
```bash
curl http://localhost:5000/api/model-results
```

## 🛠️ Technology Stack

### Backend
- **Python 3.8+** - Programming language
- **Flask** - Web framework
- **Scikit-learn** - Machine learning library
- **NLTK** - Natural language processing
- **Pandas & NumPy** - Data manipulation
- **Joblib** - Model serialization

### Machine Learning
- **Naive Bayes** - Probabilistic classifier
- **Logistic Regression** - Linear classifier
- **Support Vector Machine (SVM)** - Kernel-based classifier
- **Random Forest** - Ensemble method
- **TF-IDF** - Feature extraction

### Frontend
- **HTML5 & CSS3** - Structure and styling
- **JavaScript** - Interactivity
- **Chart.js** - Interactive charts
- **Font Awesome** - Icons

### Visualization
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations

## 📊 Model Performance

| Model | Training Accuracy | Testing Accuracy | CV Score |
|-------|------------------|------------------|----------|
| Naive Bayes | 99.1% | 97.0% | 96.8% |
| Logistic Regression | 99.5% | 96.7% | 96.5% |
| **SVM** | **99.8%** | **98.6%** | **98.3%** |
| Random Forest | 99.9% | 97.8% | 97.5% |

**Best Model: SVM (Support Vector Machine)**
- Test Accuracy: 98.57%
- Ham Precision: 98%
- Spam Precision: 100%
- Overall F1-Score: 99%

## 🔍 How It Works

### 1. Text Preprocessing
```
Original: "FREE! Click HERE to win $1000!!!"
          ↓
Lowercase: "free! click here to win $1000!!!"
          ↓
Remove Special Chars: "free click here to win"
          ↓
Remove Stopwords: "free click win"
          ↓
Stemming: "free click win"
```

### 2. Feature Extraction (TF-IDF)
- Converts text to numerical features
- Assigns weights based on term frequency and importance
- Creates a 3,000-dimensional feature vector

### 3. Classification
- Pre-trained model analyzes features
- Calculates probability scores
- Returns prediction with confidence

## 🎯 Use Cases

- **Email Filtering** - Automatically filter spam from inbox
- **Message Classification** - Categorize SMS/messages
- **Content Moderation** - Flag unwanted content
- **Security Systems** - Detect phishing attempts
- **Educational Tool** - Learn ML and NLP concepts


## 📄 License

This project is for educational purposes. Dataset credit: SMS Spam Collection.

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Improve documentation
- Submit pull requests

## 📧 Contact

For questions or support, please create an issue in the repository.

---

## 🎓 Learning Outcomes

By exploring this project, you'll learn:

1. **Machine Learning Pipeline** - From data preprocessing to deployment
2. **Text Classification** - NLP techniques for text analysis
3. **Model Comparison** - Evaluating different ML algorithms
4. **Web Development** - Building interactive Flask applications
5. **Data Visualization** - Creating insightful charts and graphs
6. **REST APIs** - Designing API endpoints for ML models
7. **UI/UX Design** - Modern web interface design

---

**Made with ❤️ using Python, Flask & Machine Learning**

🌟 Star this project if you found it helpful!
