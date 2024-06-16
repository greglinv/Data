#20,000 newsgroup documents, partitioned across 20 different newsgroups
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Fetch the dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups_data.data
labels = newsgroups_data.target

# Convert the collection of text documents to a matrix of token counts
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(documents)

print("Data shape:", X.shape)
