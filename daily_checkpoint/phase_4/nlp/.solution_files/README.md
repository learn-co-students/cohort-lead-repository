## Natural Language Processing

In this exercise we will attempt to classify text messages as "SPAM" or "HAM" using TF-IDF Vectorization. Once we successfully classify our texts we will examine our results to see which words are most important to each class of text messages. 

Complete the functions below and answer the question(s) at the end. 


```python
# import necessary libraries 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
```


```python
# read in data
df_messages = pd.read_csv('data/spam.csv', usecols=[0,1])

# convert string labels to 1 or 0 
le = LabelEncoder()
df_messages['target'] = le.fit_transform(df_messages['v1'])

# examine our data
df_messages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### TF-IDF


```python
# separate features and labels 
X = df_messages['v2']
y = df_messages['target']

# generate a list of stopwords for TfidfVectorizer to ignore
stopwords_list = stopwords.words('english') + list(string.punctuation)
```

<b>1) Let's create a function that takes in our various texts along with their respective labels and uses TF-IDF to vectorize the texts.  Recall that TF-IDF helps us "vectorize" text (turn text into numbers) so we can do "math" with it.  It is used to reflect how relevant a term is in a given document in a numerical way. </b>


```python
# generate tf-idf vectorization (use sklearn's TfidfVectorizer) for our data
def tfidf(X, y,  stopwords_list): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    vectorizer = TfidfVectorizer(stop_words=stopwords_list)
    tf_idf_train = vectorizer.fit_transform(X_train)
    tf_idf_test = vectorizer.transform(X_test)
    return tf_idf_train, tf_idf_test, y_train, y_test, vectorizer
```


```python
tf_idf_train, tf_idf_test, y_train, y_test, vectorizer = tfidf(X, y, stopwords_list)
```

### Classification

<b>2) Now that we have a set of vectorized training data we can use this data to train a classifier to learn how to classify a specific text based on the vectorized version of the text. Below we have initialized a simple Naive Bayes Classifier and Random Forest Classifier. Complete the function below which will accept a classifier object, a vectorized training set, vectorized test set, and list of training labels and return a list of predictions for our training set and a separate list of predictions for our test set.</b> 


```python
nb_classifier = MultinomialNB()
rf_classifier = RandomForestClassifier(n_estimators=100)
```


```python
# create a function that takes in a classifier and trains it on our tf-idf vectors and generates test and train predictiions
def classify_text(classifier, tf_idf_train, tf_idf_test, y_train):
    classifier.fit(tf_idf_train, y_train)
    train_preds = classifier.predict(tf_idf_train)
    test_preds = classifier.predict(tf_idf_test)
    return train_preds, test_preds
```


```python
# generate predictions with Naive Bayes Classifier
nb_train_preds, nb_test_preds = classify_text(nb_classifier, tf_idf_train, tf_idf_test, y_train)

# evaluate performance of Naive Bayes Classifier
print(confusion_matrix(y_test, nb_test_preds))
print(accuracy_score(y_test, nb_test_preds))
```

    [[1202    0]
     [  44  147]]
    0.968413496051687



```python
# generate predictions with Random Forest Classifier
rf_train_preds, rf_test_preds = classify_text(rf_classifier, tf_idf_train, tf_idf_test, y_train)

# evaluate performance of Random Forest Classifier
print(confusion_matrix(y_test, rf_test_preds))
print(accuracy_score(y_test, rf_test_preds))
```

    [[1201    1]
     [  32  159]]
    0.9763101220387652


You can see both classifiers do a pretty good job classifying texts as either "SPAM" or "HAM". Let's figure out which words are the most important to each class of texts! Recall that Inverse Document Frequency can help us determine which words are most important in an entire corpus or group of documents. 

<b>3) Create a function that calculates the inverse document frequency (IDF) of each word in our collection of texts.</b>


```python
def get_idf(class_, df, stopwords_list):
    docs = df[df.v1==class_].v2
    class_dict = {} 
    for doc in docs:
        words = set(doc.split())
        for word in words:
            if word.lower() not in stopwords_list: 
                class_dict[word.lower()] = class_dict.get(word.lower(), 0) + 1
    idf_df = pd.DataFrame.from_dict(class_dict, orient='index')
    idf_df.columns = ['IDF']
    idf_df.IDF = np.log(len(docs)/idf_df.IDF)
    idf_df = idf_df.sort_values(by="IDF", ascending=True)
    return idf_df.head(10)
```


```python
get_idf('spam', df_messages, stopwords_list)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IDF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>call</th>
      <td>2.277439</td>
    </tr>
    <tr>
      <th>free</th>
      <td>4.698113</td>
    </tr>
    <tr>
      <th>txt</th>
      <td>5.574627</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.746154</td>
    </tr>
    <tr>
      <th>ur</th>
      <td>6.073171</td>
    </tr>
    <tr>
      <th>u</th>
      <td>7.047170</td>
    </tr>
    <tr>
      <th>text</th>
      <td>7.114286</td>
    </tr>
    <tr>
      <th>mobile</th>
      <td>7.114286</td>
    </tr>
    <tr>
      <th>claim</th>
      <td>7.182692</td>
    </tr>
    <tr>
      <th>reply</th>
      <td>7.781250</td>
    </tr>
  </tbody>
</table>
</div>




```python
get_idf('ham', df_messages, stopwords_list)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IDF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>u</th>
      <td>7.310606</td>
    </tr>
    <tr>
      <th>i'm</th>
      <td>14.067055</td>
    </tr>
    <tr>
      <th>get</th>
      <td>17.481884</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20.188285</td>
    </tr>
    <tr>
      <th>got</th>
      <td>22.133028</td>
    </tr>
    <tr>
      <th>go</th>
      <td>22.133028</td>
    </tr>
    <tr>
      <th>&amp;lt;#&amp;gt;</th>
      <td>22.546729</td>
    </tr>
    <tr>
      <th>call</th>
      <td>23.309179</td>
    </tr>
    <tr>
      <th>like</th>
      <td>23.309179</td>
    </tr>
    <tr>
      <th>come</th>
      <td>23.651961</td>
    </tr>
  </tbody>
</table>
</div>



### Explain
<b> 4) Imagine that the word "school" has the highest TF-IDF value in the second document of our test data. What does that tell us about the word school? </b>


```python
# Answer: The word "school" is very unique. It is not found frequently across 
# many documents but its present in the second document meaning it has significant importance to this document.
```
