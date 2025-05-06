import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset (SMS Spam Collection)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=['label', 'message'])

# Encode labels: spam = 1, ham = 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Basic preprocessing
def preprocess(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['message'] = df['message'].apply(preprocess)

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict on new input
def predict_spam(text):
    processed = preprocess(text)
    vect = vectorizer.transform([processed])
    pred = model.predict(vect)
    return "Spam" if pred[0] == 1 else "Ham"

# Example
data = pd.read_csv("./spam_ham_dataset.csv")
# Preprocess the data
emails = data["text"].array
for i in emails:
    print(predict_spam(i))


