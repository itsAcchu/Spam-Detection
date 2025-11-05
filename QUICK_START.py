"""
QUICK START GUIDE - Email Spam Detection Web Application
=========================================================

Follow these simple steps to get your spam detection system up and running!
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║               🛡️  EMAIL SPAM DETECTION SYSTEM - SETUP GUIDE                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 PREREQUISITES:
   ✓ Python 3.8 or higher installed
   ✓ pip package manager
   ✓ spam.csv file in the project folder

📁 PROJECT STRUCTURE:
   Your folder should look like this:
   
   Email-Spam-Detection/
   ├── spam.csv                     ← Your dataset (must have this!)
   ├── train_and_save_model.py      ← Model training script
   ├── app.py                       ← Flask web application
   ├── requirements.txt             ← Python packages list
   ├── README.md                    ← Documentation
   └── templates/
       └── index.html               ← Web interface

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 STEP 1: INSTALL REQUIRED PACKAGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Open your terminal/command prompt in the project folder and run:

   Windows:
   --------
   pip install flask scikit-learn pandas numpy matplotlib seaborn nltk joblib

   OR use the requirements file:
   pip install -r requirements.txt

   Linux/Mac:
   ----------
   pip3 install flask scikit-learn pandas numpy matplotlib seaborn nltk joblib

   OR with --break-system-packages flag if needed:
   pip install -r requirements.txt --break-system-packages

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 STEP 2: TRAIN THE MODELS (First Time Only - Takes ~2 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run the training script:

   python train_and_save_model.py

This will:
   ✓ Load and analyze the spam.csv dataset (5,572 emails)
   ✓ Preprocess all text data
   ✓ Train 4 different ML models (Naive Bayes, Logistic Regression, SVM, Random Forest)
   ✓ Compare their performance
   ✓ Save the best model
   ✓ Generate beautiful visualizations
   ✓ Create all necessary data files

Expected output:
   ================================================================================
   LOADING DATASET
   ================================================================================
   ✓ Dataset loaded: 5,572 emails
   
   ================================================================================
   TEXT PREPROCESSING
   ================================================================================
   ✓ Text preprocessing complete
   
   ================================================================================
   MODEL TRAINING AND EVALUATION
   ================================================================================
   Training Naive Bayes...
     ✓ Training Accuracy: 0.9910
     ✓ Testing Accuracy: 0.9704
   
   Training Logistic Regression...
     ✓ Training Accuracy: 0.9952
     ✓ Testing Accuracy: 0.9668
   
   Training SVM...
     ✓ Training Accuracy: 0.9979
     ✓ Testing Accuracy: 0.9857
   
   Training Random Forest...
     ✓ Training Accuracy: 0.9990
     ✓ Testing Accuracy: 0.9776
   
   🏆 Best Model: SVM
      Test Accuracy: 0.9857
   
   ✓ All models, vectorizer, and data saved successfully!

After this step, you'll have these new folders:
   ├── models/              ← Trained models
   ├── data/                ← Statistics and metrics
   └── static/images/       ← Visualization charts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌐 STEP 3: START THE WEB APPLICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run the Flask app:

   python app.py

You should see:
   ================================================================================
   EMAIL SPAM DETECTION - WEB APPLICATION
   ================================================================================
   
   ✓ Models loaded successfully!
   
   🚀 Starting Flask server...
   ================================================================================
   
   📱 Access the application at: http://localhost:5000
   
   ================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎉 STEP 4: OPEN YOUR BROWSER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Open your web browser and go to:

   http://localhost:5000

You'll see a beautiful interface with 5 tabs:

   1. 🤖 PREDICT    - Test your emails in real-time
   2. 📊 DATASET    - View dataset statistics
   3. 🧠 MODELS     - Compare model performance
   4. 📈 ANALYSIS   - Interactive charts and metrics
   5. ℹ️  ABOUT     - Project information

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 QUICK TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Try these test emails:

   SPAM Example:
   "Congratulations! You've won a $1000 gift card. Click here to claim now!"
   
   Expected Result: ⚠️ SPAM DETECTED! (95%+ confidence)

   HAM Example:
   "Hey, are we still meeting for lunch tomorrow?"
   
   Expected Result: ✅ LEGITIMATE EMAIL (95%+ confidence)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🐛 TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: "FileNotFoundError: spam.csv"
Solution: Make sure spam.csv is in the same folder as the scripts!

Problem: "No module named 'flask'"
Solution: Install packages again: pip install flask scikit-learn pandas numpy matplotlib seaborn nltk joblib

Problem: "Models not found"
Solution: Run train_and_save_model.py first!

Problem: Port 5000 already in use
Solution: Change the port in app.py:
   app.run(debug=True, host='0.0.0.0', port=8080)
   Then access: http://localhost:8080

Problem: NLTK stopwords error
Solution: Run Python and execute:
   import nltk
   nltk.download('stopwords')

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 FEATURES OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ What You Can Do:

   1. Real-time Email Classification
      • Type or paste any email
      • Get instant spam/ham prediction
      • View confidence scores and probabilities

   2. Dataset Exploration
      • See total email count (5,572)
      • View ham vs spam distribution
      • Analyze text length patterns

   3. Model Performance
      • Compare 4 ML algorithms
      • View confusion matrices
      • See detailed metrics (precision, recall, F1-score)

   4. Interactive Visualizations
      • Beautiful gradient charts
      • Real-time updates
      • Export-ready images

   5. Example Testing
      • One-click example emails
      • Both spam and ham samples
      • Clear visual feedback

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 EXPECTED RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your models should achieve:

   Model Performance (Typical):
   ┌────────────────────────┬──────────────────┐
   │ Naive Bayes            │ 97.04% accuracy  │
   │ Logistic Regression    │ 96.68% accuracy  │
   │ SVM (Best)             │ 98.57% accuracy  │
   │ Random Forest          │ 97.76% accuracy  │
   └────────────────────────┴──────────────────┘

   Best Model Metrics (SVM):
   • Overall Accuracy: 98.57%
   • Ham Precision: 98%
   • Spam Precision: 100%
   • Ham Recall: 100%
   • Spam Recall: 89%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎓 LEARNING OUTCOMES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

By using this project, you'll understand:

   ✓ Machine Learning pipeline (preprocessing → training → evaluation)
   ✓ Text classification with NLP
   ✓ Model comparison and selection
   ✓ Building web applications with Flask
   ✓ Creating interactive visualizations
   ✓ REST API design
   ✓ Modern UI/UX principles

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📞 NEED HELP?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   • Read the full README.md for detailed documentation
   • Check the troubleshooting section above
   • Verify all files are in the correct locations
   • Make sure spam.csv is present

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎉 THAT'S IT! YOU'RE READY TO GO!

   1. pip install -r requirements.txt
   2. python train_and_save_model.py
   3. python app.py
   4. Open http://localhost:5000

Happy spam detecting! 🛡️

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Made with ❤️ using Python, Flask & Machine Learning
""")
