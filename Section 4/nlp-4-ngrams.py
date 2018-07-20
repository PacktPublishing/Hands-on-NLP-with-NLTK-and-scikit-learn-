import collections
import nltk
import os
from sklearn import (
    datasets, model_selection, feature_extraction, linear_model, naive_bayes,
    ensemble
)


def extract_features(corpus):
    '''Extract TF-IDF features from corpus'''

    sa_stop_words = nltk.corpus.stopwords.words("english")

    # words that might invert a sentence's meaning
    white_list = [
        'what', 'but', 'if', 'because', 'as', 'until', 'against',
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'why', 'how', 'all', 'any',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'can', 'will', 'just', 'don', 'should']

    # take these out of the standard NLTK stop word list
    sa_stop_words = [sw for sw in sa_stop_words if sw not in white_list]

    # vectorize means we turn non-numerical data into an array of numbers
    count_vectorizer = feature_extraction.text.CountVectorizer(
        lowercase=True,  # for demonstration, True by default
        tokenizer=nltk.word_tokenize,  # use the NLTK tokenizer
        min_df=2,  # minimum document frequency, i.e. the word must appear more than once.
        ngram_range=(1, 2),
        stop_words=sa_stop_words
    )
    processed_corpus = count_vectorizer.fit_transform(corpus)
    processed_corpus = feature_extraction.text.TfidfTransformer().fit_transform(
        processed_corpus)

    return processed_corpus


data_directory = 'movie_reviews'
movie_sentiment_data = datasets.load_files(data_directory, shuffle=True)
print('{} files loaded.'.format(len(movie_sentiment_data.data)))
print('They contain the following classes: {}.'.format(
    movie_sentiment_data.target_names))

movie_tfidf = extract_features(movie_sentiment_data.data)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    movie_tfidf, movie_sentiment_data.target, test_size=0.30, random_state=42)

# similar to nltk.NaiveBayesClassifier.train()
clf1 = linear_model.LogisticRegression()
clf1.fit(X_train, y_train)
print('Logistic Regression performance: {}'.format(clf1.score(X_test, y_test)))

clf2 = linear_model.SGDClassifier()
clf2.fit(X_train, y_train)
print('SGDClassifier performance: {}'.format(clf2.score(X_test, y_test)))

clf3 = naive_bayes.MultinomialNB()
clf3.fit(X_train, y_train)
print('MultinomialNB performance: {}'.format(clf3.score(X_test, y_test)))

clf4 = naive_bayes.BernoulliNB()
clf4.fit(X_train, y_train)
print('BernoulliNB performance: {}'.format(clf4.score(X_test, y_test)))


voting_model = ensemble.VotingClassifier(
    estimators=[('lr', clf1), ('sgd', clf2), ('mnb', clf3), ('bnb', clf4)],
    voting='hard')
voting_model.fit(X_train, y_train)
print('Voting classifier performance: {}'.format(
    voting_model.score(X_test, y_test)))

