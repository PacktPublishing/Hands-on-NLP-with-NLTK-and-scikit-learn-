from sklearn import feature_extraction


corpus = [
    'Convert a collection of text documents to a matrix of token occurrences',
    'It turns a collection of text documents into a scipy.sparse matrix holding token occurrence counts (or binary occurrence information), possibly normalized as token frequencies if norm=’l1’ or projected on the euclidean unit sphere if norm=’l2’.',
    'This text vectorizer implementation uses the hashing trick to find the token string name to feature integer index mapping.',
    'This strategy has several advantages:',
    'it is very low memory scalable to large datasets as there is no need to store a vocabulary dictionary in memory',
    'it is fast to pickle and un-pickle as it holds no state besides the constructor parameters',
    'it can be used in a streaming (partial fit) or parallel pipeline as there is no state computed during fit.'
]

print('Processing corpus: {} documents'.format(len(corpus)))

print('Count Vectorizer:\n')
vectorizer = feature_extraction.text.CountVectorizer()
X = vectorizer.fit_transform(corpus)
# Count Vectorizer stores a dictionary: a number per word
print(vectorizer.vocabulary_)
print('Resulting matrix has {} data points and {} features.\n'.format(
    X.shape[0], X.shape[1]))
print('Document 1: \n{}'.format(X[0].toarray()))
# as the number of words increase, you need a bigger and bigger dictionary!


print('Hashing Vectorizer:\n')

# norm=None means we don't normalize the values
# alternative_sign=False means that we don't alternate the value's signs to
#   conserve any mathematical properties
vectorizer = feature_extraction.text.HashingVectorizer(
    norm=None, alternate_sign=False)
X = vectorizer.transform(corpus)  # not fit_transform

print('Resulting matrix has {} data points and {} features.\n'.format(
    X.shape[0], X.shape[1]))

# > Resulting matrix has 7 data points and 1048576 features.

print('Document 1: \n{}'.format(X[0]))

# Document 1: 
#   (0, 22468)	0.2886751345948129
#   (0, 124863)	-0.2886751345948129
#   (0, 164975)	-0.2886751345948129
#   (0, 174171)	0.2886751345948129
#   (0, 264705)	0.2886751345948129
#   (0, 479532)	0.5773502691896258
#   (0, 548700)	-0.2886751345948129
#   (0, 676585)	-0.2886751345948129
#   (0, 741852)	-0.2886751345948129
#   Read the above as:
#   (document_index, feature_index) 
