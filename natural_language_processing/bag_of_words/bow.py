# Multiple Linear Regression

# Importing the libraries
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Importing the dataset, tab separated, ignore " characters
dataset = pd.read_csv('../../data/Restaurant_Reviews.tsv', delimiter="\t", quoting=3)

# Clean the dataset to ease the learning process

# download stopwords # Remove non-relevant words from reviews "The" "a" "i" "an" "them" "they" ect.
nltk.download('stopwords')
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

corpus = []
for i in range(0, 1000):
    # initial cleaning removes non letter characters
    cleaned_review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # Make letters lowercase
    cleaned_review = cleaned_review.lower()

    words = cleaned_review.split()
    # Split string so that we can apply stemming.
    #  Stemming converts words like "Loved" to "Love". Removed conjugation to simplify text
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if word not in set(all_stopwords)]

    cleaned_review = ' '.join(words)
    corpus.append(cleaned_review)

print(corpus)

# tokenize the words in the corpus

# 171,476 words in english language, most adult speakers range from 20-35k words in their vocab.
# Out of those humans use just 3k words provides coverage for 95% of common texts

count_vectorizer = CountVectorizer(max_features=1500)  # decided by looking at the values after running the CV once
x = count_vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# print(len(x[0])) = 1566 words after tokenization

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

classifier = naive_bayes.GaussianNB()
classifier.fit(x_train, y_train)

y_predicted = classifier.predict(x_test)

print(y_predicted)

# Evaluating the Model Performance
confusion_m = confusion_matrix(y_test, y_predicted)

print(confusion_m)
print(f'accuracy {accuracy_score(y_test, y_predicted)}')

ConfusionMatrixDisplay(confusion_m).plot()
pyplot.show()