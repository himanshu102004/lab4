from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score

# Download the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)
y_pred_nb = nb_classifier.predict(X_test_tfidf)
f1_score_nb = f1_score(y_test, y_pred_nb, average='weighted')
report_nb = classification_report(y_test, y_pred_nb, target_names=newsgroups.target_names)
print("Naive Bayes Classifier Report:")
print(report_nb)
print("F1-Score for Naive Bayes: ", f1_score_nb)

# Rocchio Classifier
rocchio_classifier = NearestCentroid()
rocchio_classifier.fit(X_train_tfidf, y_train)
y_pred_rocchio = rocchio_classifier.predict(X_test_tfidf)
f1_score_rocchio = f1_score(y_test, y_pred_rocchio, average='weighted')
report_rocchio = classification_report(y_test, y_pred_rocchio, target_names=newsgroups.target_names)
print("Rocchio Classifier Report:")
print(report_rocchio)
print("F1-Score for Rocchio: ", f1_score_rocchio)

# K-Nearest Neighbor Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
knn_classifier.fit(X_train_tfidf, y_train)
y_pred_knn = knn_classifier.predict(X_test_tfidf)
f1_score_knn = f1_score(y_test, y_pred_knn, average='weighted')
report_knn = classification_report(y_test, y_pred_knn, target_names=newsgroups.target_names)
print("K-Nearest Neighbor Classifier Report:")
print(report_knn)
print("F1-Score for K-Nearest Neighbor: ", f1_score_knn)
