# Similar_Question_Indentification

📌Problem Statement:
Given two questions, determine whether they are semantically duplicate or not.
This is a binary classification problem:
1 → Duplicate,
0 → Not Duplicate.
Dataset used: Quora Question Pairs Dataset

🧠 Project Pipeline

1️⃣ Data Preprocessing:
Lowercasing,
Removing HTML tags,
Removing punctuation,
Expanding contractions (e.g., "can't" → "can not"),
Cleaning special symbols,
Handling numeric formats,

2️⃣ Feature Engineering:
This project heavily relies on handcrafted NLP features:
🔹 Basic Features:
Question length,
Number of words,
Common word count,
Word share ratio.
🔹 Token-Based Feature:
Common non-stopwords ratio,
Common stopwords ratio,
Common token ratio,
First word equality,
Last word equality.
🔹 Length-Based Features:
Absolute length difference,
Mean token length,
Longest common substring ratio.
🔹 Fuzzy Matching Features:
Using fuzzywuzzy:
fuzz_ratio,
partial_ratio,
token_sort_ratio,
token_set_ratio.
🔹 Bag of Words (BoW):
CountVectorizer (max_features=3000),
Applied separately on Question1 and Question2,
Final feature dimension ≈ 6000+ features.

📊 Dimensionality Reduction:
Used t-SNE for visualization of:
15 engineered features,
2D and 3D embedding for cluster visualization.

🤖 Model Used:
Random Forest Classifier,
RandomForestClassifier()

📈 Model Performance:
Accuracy: ~78.75
Confusion Matrix:
[[3291  521]
 [ 754 1434]]
 
🏗 Final Feature Vector Size:
Engineered features: 23
BoW features: 6000
Total: 6023 features

🚀 Making Predictions:
Example:
q1 = "what is the capital of india?" and 
q2 = "where is the current capital of india?" ,
rf.predict(query_point_creator(q1,q2))

Output:
array([1])
💾 Model Deployment:
Model and vectorizer are saved using pickle:
pickle.dump(rf, open('model.pkl','wb')) ,
pickle.dump(cv, open('cv.pkl','wb'))

📦 Dependencies:
numpy,
pandas,
matplotlib,
seaborn,
scikit-learn,
nltk,
beautifulsoup4,
fuzzywuzzy,
distance,

🧩 Key Learnings:
Importance of feature engineering in NLP,
Combining statistical + fuzzy matching features,
Effect of dimensionality on classical ML models,
Visualizing clusters using t-SNE,
End-to-end ML pipeline design.

🔮 Future Improvements:
Replace BoW with TF-IDF,
Use Word2Vec / GloVe embeddings,
Try Deep Learning models (LSTM, Siamese Network),
Fine-tune BERT for semantic similarity,
Deploy using Flask / FastAPI.
