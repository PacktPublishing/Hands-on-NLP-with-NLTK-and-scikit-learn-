import itertools
import numpy as np
from sklearn import (
    datasets, feature_extraction, model_selection, pipeline,
    decomposition, preprocessing, naive_bayes
)
import matplotlib.pyplot as plt


if __name__ == '__main__':
    newsgroups_data = datasets.load_files(
        '20_newsgroups', shuffle=True, random_state=42, encoding='ISO-8859-1')

    print('Data loaded.\nClasses = {classes}\n{datapoints}'.format(
        classes=newsgroups_data.target_names, datapoints=len(newsgroups_data.data)))

    # sometimes the label is present in the training data
    print(newsgroups_data.data[0])
    # remove any label present in the features

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        newsgroups_data.data, newsgroups_data.target, test_size=0.33,
        random_state=42)

    model = pipeline.Pipeline([
        ('counts', feature_extraction.text.CountVectorizer()),
        ('tfidf', feature_extraction.text.TfidfTransformer()),
        ('SVD', decomposition.TruncatedSVD(128)),
        ('normalize', preprocessing.Normalizer(copy=False)),
        ('naivebayes', naive_bayes.GaussianNB())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(model.score(X_test, y_test))
