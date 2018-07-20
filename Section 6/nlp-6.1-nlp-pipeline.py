import nltk


corpus = [
    """
    This strategy has several advantages:
    it is very low memory scalable to large datasets as there is no need to store a vocabulary dictionary in memory
    it is fast to pickle and un-pickle as it holds no state besides the constructor parameters
    it can be used in a streaming (partial fit) or parallel pipeline as there is no state computed during fit.
    """,
    """
    It turns a collection of text documents into a scipy.sparse matrix holding token occurrence counts (or binary occurrence information), 
    possibly normalized as token frequencies if norm=’l1’ or projected on the euclidean unit sphere if norm=’l2’.
    """
]


def pipeline(f):
    '''pipeline decorator that calls next() on function f()'''
    def start_pipeline(*args, **kwargs):
        nf = f(*args, **kwargs)
        next(nf)
        return nf
    return start_pipeline


def ingest(corpus, targets):
    for text in corpus:
        for t in targets:
            t.send(text)


@pipeline
def tokenize_sentences(targets):
    while True:
        text = (yield)  # (yield) gets an item from an upstream step
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            for target in targets:
                target.send(sentence)  # send() sends data downstream


@pipeline
def tokenize_words(targets):
    while True:
        sentence = (yield)
        words = nltk.word_tokenize(sentence)
        for target in targets:
            target.send(words)


@pipeline
def pos_tagging(targets):
    while True:
        words = (yield)
        tagged_words = nltk.pos_tag(words)

        for target in targets:
            target.send(tagged_words)


@pipeline
def ne_chunking(targets):
    while True:
        tagged_words = (yield)
        ner_tagged = nltk.ne_chunk(tagged_words)
        for target in targets:
            target.send(ner_tagged)


@pipeline
def printline(title):
    while True:
        line = (yield)
        print(title)
        print(line)

ingest(corpus, [
    tokenize_sentences([
        tokenize_words([
            printline('Word tokens:'),
            pos_tagging([
                ne_chunking([
                    printline('Results:')
                ])
            ])
        ])
    ])
])
