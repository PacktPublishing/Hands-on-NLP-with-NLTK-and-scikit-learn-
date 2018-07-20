import itertools
import numpy as np
import nltk
from sklearn import (
    datasets, feature_extraction, model_selection, pipeline,
    naive_bayes, metrics
)
import matplotlib.pyplot as plt


def extract_features(corpus):
    '''Extract TF-IDF features from corpus'''

    stop_words = nltk.corpus.stopwords.words("english")

    # vectorize means we turn non-numerical data into an array of numbers
    count_vectorizer = feature_extraction.text.CountVectorizer(
        lowercase=True,  # for demonstration, True by default
        tokenizer=nltk.word_tokenize,  # use the NLTK tokenizer
        min_df=2,  # minimum document frequency, i.e. the word must appear more than once.
        ngram_range=(1, 2),
        stop_words=stop_words
    )
    processed_corpus = count_vectorizer.fit_transform(corpus)
    processed_corpus = feature_extraction.text.TfidfTransformer().fit_transform(
        processed_corpus)

    return processed_corpus

if __name__ == '__main__':
    newsgroups_data = datasets.load_files(
        '20_newsgroups', shuffle=True, random_state=42, encoding='ISO-8859-1')

    print('Data loaded.\nClasses = {classes}\n{datapoints}'.format(
        classes=newsgroups_data.target_names,
        datapoints=len(newsgroups_data.data)))

    print(newsgroups_data.data[0])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        newsgroups_data.data, newsgroups_data.target, test_size=0.33,
        random_state=42)

    stop_words = nltk.corpus.stopwords.words("english")

    model = pipeline.Pipeline([
        ('counts', feature_extraction.text.CountVectorizer(
            lowercase=True,  # for demonstration, True by default
            tokenizer=nltk.word_tokenize,  # use the NLTK tokenizer
            min_df=2,  # minimum document frequency, i.e. the word must appear more than once.
            ngram_range=(1, 2),
            stop_words=stop_words
        )),
        ('tfidf', feature_extraction.text.TfidfTransformer()),
        ('naivebayes', naive_bayes.MultinomialNB()),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print('Accuracy of multinomial naive bayes= {}'.format(
        np.mean(y_pred == y_test)))

    print(metrics.classification_report(
        y_test, y_pred, target_names=newsgroups_data.target_names))

    grid_search_model = model_selection.GridSearchCV(
        model,
        {
            'counts__ngram_range': [(1, 1), (1, 2)],
            'naivebayes__alpha': (0.1, 3.0)
        },
        n_jobs=-1  # detect how many cores are installed and uses them all
    )

    grid_search_model.fit(X_train, y_train)
    print(grid_search_model.cv_results_)
