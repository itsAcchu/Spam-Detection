"""
Enhanced Email Spam Detection - Model Training and Data Generation
Trains models, saves them, and generates all data for frontend visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import warnings
import joblib
import json
import os
warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('static/images', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download('stopwords', quiet=True)
print("✓ NLTK data downloaded successfully!\n")

# ============================================================================
# 1. LOAD THE DATASET
# ============================================================================
print("="*80)
print("STEP 1: LOADING DATASET")
print("="*80)

df = pd.read_csv('spam.csv', encoding='latin-1')
print(f"✓ Dataset loaded: {df.shape[0]} emails")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("STEP 2: DATA PREPROCESSING")
print("="*80)

df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df = df.dropna()

# Calculate statistics for frontend
ham_count = (df['label'] == 'ham').sum()
spam_count = (df['label'] == 'spam').sum()
total_count = len(df)

print(f"✓ Ham emails: {ham_count} ({ham_count/total_count*100:.1f}%)")
print(f"✓ Spam emails: {spam_count} ({spam_count/total_count*100:.1f}%)")

# Save dataset statistics
dataset_stats = {
    'total_emails': int(total_count),
    'ham_count': int(ham_count),
    'spam_count': int(spam_count),
    'ham_percentage': round(ham_count/total_count*100, 2),
    'spam_percentage': round(spam_count/total_count*100, 2)
}

with open('data/dataset_stats.json', 'w') as f:
    json.dump(dataset_stats, f, indent=2)

# Visualization 1: Label Distribution (Enhanced)
plt.figure(figsize=(10, 6))
colors = ['#2ecc71', '#e74c3c']
labels = df['label'].value_counts()
explode = (0.05, 0.05)

plt.subplot(1, 2, 1)
plt.pie(labels.values, labels=['Ham', 'Spam'], autopct='%1.1f%%', 
        colors=colors, explode=explode, shadow=True, startangle=90)
plt.title('Email Distribution', fontsize=14, fontweight='bold')

plt.subplot(1, 2, 2)
bars = plt.bar(['Ham', 'Spam'], labels.values, color=colors, edgecolor='black', linewidth=2)
plt.ylabel('Count', fontsize=12)
plt.title('Email Count', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('static/images/dataset_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Dataset distribution chart saved")

# ============================================================================
# 3. TEXT PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: TEXT PREPROCESSING")
print("="*80)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

print("Processing messages...")
df['processed_message'] = df['message'].apply(preprocess_text)
print("✓ Text preprocessing complete")

# Calculate text statistics
df['message_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

text_stats = {
    'avg_length_ham': float(df[df['label']=='ham']['message_length'].mean()),
    'avg_length_spam': float(df[df['label']=='spam']['message_length'].mean()),
    'avg_words_ham': float(df[df['label']=='ham']['word_count'].mean()),
    'avg_words_spam': float(df[df['label']=='spam']['word_count'].mean())
}

with open('data/text_stats.json', 'w') as f:
    json.dump(text_stats, f, indent=2)

# Visualization 2: Text Length Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Message length distribution
axes[0, 0].hist(df[df['label']=='ham']['message_length'], bins=50, alpha=0.7, 
                label='Ham', color='#2ecc71', edgecolor='black')
axes[0, 0].hist(df[df['label']=='spam']['message_length'], bins=50, alpha=0.7, 
                label='Spam', color='#e74c3c', edgecolor='black')
axes[0, 0].set_xlabel('Message Length (characters)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Message Length Distribution', fontweight='bold')
axes[0, 0].legend()

# Word count distribution
axes[0, 1].hist(df[df['label']=='ham']['word_count'], bins=50, alpha=0.7, 
                label='Ham', color='#2ecc71', edgecolor='black')
axes[0, 1].hist(df[df['label']=='spam']['word_count'], bins=50, alpha=0.7, 
                label='Spam', color='#e74c3c', edgecolor='black')
axes[0, 1].set_xlabel('Word Count')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Word Count Distribution', fontweight='bold')
axes[0, 1].legend()

# Average length comparison
categories = ['Ham', 'Spam']
avg_lengths = [text_stats['avg_length_ham'], text_stats['avg_length_spam']]
bars = axes[1, 0].bar(categories, avg_lengths, color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=2)
axes[1, 0].set_ylabel('Average Length (characters)')
axes[1, 0].set_title('Average Message Length', fontweight='bold')
for bar in bars:
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontweight='bold')

# Average word count comparison
avg_words = [text_stats['avg_words_ham'], text_stats['avg_words_spam']]
bars = axes[1, 1].bar(categories, avg_words, color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=2)
axes[1, 1].set_ylabel('Average Word Count')
axes[1, 1].set_title('Average Words per Message', fontweight='bold')
for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('static/images/text_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Text analysis chart saved")

# ============================================================================
# 4. FEATURE EXTRACTION
# ============================================================================
print("\n" + "="*80)
print("STEP 4: FEATURE EXTRACTION")
print("="*80)

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['processed_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Testing set: {len(X_test)} samples")

tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Save vectorizer
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
print("✓ TF-IDF vectorizer saved")

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
print("\n" + "="*80)
print("STEP 5: TRAINING MULTIPLE MODELS")
print("="*80)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    
    # Training accuracy
    train_pred = model.predict(X_train_tfidf)
    train_acc = accuracy_score(y_train, train_pred)
    
    # Testing accuracy
    test_pred = model.predict(X_test_tfidf)
    test_acc = accuracy_score(y_test, test_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
    cv_mean = cv_scores.mean()
    
    results[name] = {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'cv_score': float(cv_mean),
        'cv_std': float(cv_scores.std())
    }
    
    predictions[name] = test_pred
    
    print(f"  ✓ Training Accuracy: {train_acc:.4f}")
    print(f"  ✓ Testing Accuracy: {test_acc:.4f}")
    print(f"  ✓ CV Score: {cv_mean:.4f} (+/- {cv_scores.std():.4f})")

# Save results
with open('data/model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Find best model
best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")

# Save best model
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(stemmer, 'models/stemmer.pkl')
print(f"✓ Best model ({best_model_name}) saved")

# ============================================================================
# 6. MODEL COMPARISON VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: GENERATING VISUALIZATIONS")
print("="*80)

# Visualization 3: Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Test Accuracy Comparison
model_names = list(results.keys())
test_accuracies = [results[m]['test_accuracy'] for m in model_names]
colors_palette = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

bars = axes[0, 0].bar(model_names, test_accuracies, color=colors_palette, 
                       edgecolor='black', linewidth=2)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title('Model Test Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylim([0.9, 1.0])
axes[0, 0].tick_params(axis='x', rotation=15)

for bar, acc in zip(bars, test_accuracies):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

# Train vs Test Accuracy
train_accs = [results[m]['train_accuracy'] for m in model_names]
x = np.arange(len(model_names))
width = 0.35

bars1 = axes[0, 1].bar(x - width/2, train_accs, width, label='Train', 
                        color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = axes[0, 1].bar(x + width/2, test_accuracies, width, label='Test', 
                        color='#2ecc71', edgecolor='black', linewidth=1.5)

axes[0, 1].set_ylabel('Accuracy', fontsize=12)
axes[0, 1].set_title('Training vs Testing Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(model_names, rotation=15)
axes[0, 1].legend()
axes[0, 1].set_ylim([0.9, 1.0])

# Cross-Validation Scores
cv_scores = [results[m]['cv_score'] for m in model_names]
cv_stds = [results[m]['cv_std'] for m in model_names]

bars = axes[1, 0].bar(model_names, cv_scores, color=colors_palette, 
                       edgecolor='black', linewidth=2, yerr=cv_stds, capsize=5)
axes[1, 0].set_ylabel('CV Score', fontsize=12)
axes[1, 0].set_title('Cross-Validation Scores (5-Fold)', fontsize=14, fontweight='bold')
axes[1, 0].set_ylim([0.9, 1.0])
axes[1, 0].tick_params(axis='x', rotation=15)

for bar, cv in zip(bars, cv_scores):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{cv:.4f}', ha='center', va='bottom', fontweight='bold')

# Performance Summary Table
axes[1, 1].axis('tight')
axes[1, 1].axis('off')

table_data = []
for name in model_names:
    table_data.append([
        name,
        f"{results[name]['train_accuracy']:.4f}",
        f"{results[name]['test_accuracy']:.4f}",
        f"{results[name]['cv_score']:.4f}"
    ])

table = axes[1, 1].table(cellText=table_data,
                          colLabels=['Model', 'Train Acc', 'Test Acc', 'CV Score'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.3, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best model
best_idx = model_names.index(best_model_name) + 1
for i in range(4):
    table[(best_idx, i)].set_facecolor('#2ecc71')
    table[(best_idx, i)].set_text_props(weight='bold')

axes[1, 1].set_title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('static/images/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Model comparison chart saved")

# Visualization 4: Confusion Matrices for All Models
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, (name, pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'],
                ax=axes[idx], cbar_kws={'label': 'Count'})
    
    axes[idx].set_ylabel('Actual', fontsize=11)
    axes[idx].set_xlabel('Predicted', fontsize=11)
    axes[idx].set_title(f'{name}\nAccuracy: {results[name]["test_accuracy"]:.4f}', 
                        fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('static/images/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrices saved")

# Visualization 5: Detailed metrics for best model
best_pred = predictions[best_model_name]
cm = confusion_matrix(y_test, best_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'],
            ax=axes[0], cbar_kws={'label': 'Count'},
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[0].set_title(f'Best Model: {best_model_name}\nConfusion Matrix', 
                  fontsize=14, fontweight='bold')

# Classification metrics
report = classification_report(y_test, best_pred, target_names=['Ham', 'Spam'], output_dict=True)

metrics_data = [
    ['Ham', f"{report['Ham']['precision']:.3f}", f"{report['Ham']['recall']:.3f}", f"{report['Ham']['f1-score']:.3f}"],
    ['Spam', f"{report['Spam']['precision']:.3f}", f"{report['Spam']['recall']:.3f}", f"{report['Spam']['f1-score']:.3f}"],
    ['Accuracy', '', '', f"{report['accuracy']:.3f}"]
]

axes[1].axis('tight')
axes[1].axis('off')

table = axes[1].table(cellText=metrics_data,
                       colLabels=['Class', 'Precision', 'Recall', 'F1-Score'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 3)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=13)

# Style rows
for i in range(1, 4):
    for j in range(4):
        if i == 3:  # Accuracy row
            table[(i, j)].set_facecolor('#2ecc71')
            table[(i, j)].set_text_props(weight='bold')
        else:
            table[(i, j)].set_text_props(fontsize=11)

axes[1].set_title('Classification Metrics', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('static/images/best_model_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Best model metrics saved")

# Save classification report
with open('data/classification_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\n✓ Best Model: {best_model_name}")
print(f"✓ Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
print(f"\n✓ All models, vectorizer, and data saved successfully!")
print(f"✓ All visualizations generated in 'static/images/' folder")
print("\n" + "="*80)
