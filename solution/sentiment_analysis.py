import matplotlib.pyplot as plt
import requests
import re
import numpy as np
import sklearn
import html2text

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D, LSTM
import keras.optimizers
from keras.layers.embeddings import Embedding
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import spacy

from get_set_review_urls import get_review_urls

sp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words.union(set('don'))

set_review2 = "https://strategy.channelfireball.com/all-strategy/mtg/channelmagic-articles/fate-reforged-constructed-review-blue/"
set_review1 = "https://strategy.channelfireball.com/all-strategy/mtg/channelmagic-articles/theros-beyond-death-constructed-set-review-green/"
set_review3 = "https://strategy.channelfireball.com/all-strategy/mtg/channelmagic-articles/luis-scott-vargas-articles/khans-of-tarkir-constructed-review-jeskai-and-blue/"
set_review4 = "https://strategy.channelfireball.com/all-strategy/mtg/channelmagic-articles/battle-for-zendikar-constructed-set-review-red/"
set_review5 = "https://strategy.channelfireball.com/all-strategy/mtg/channelmagic-articles/battle-for-zendikar-constructed-set-review-black/"
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}


def cleanhtml(raw_html):
    """
    Removes html tags from an input string.
    :param raw_html: html content
    :return: cleaned html content, article content
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, "", raw_html)
    cleantext = " ".join(cleantext.splitlines())
    return cleantext


def preprocess_text(corp):
    """
    Preprocesses a text and returns a punctuation-less, space delimited text composed of tokens (words) composed of
    n>=2 chars.
    :param corp: text corpus
    :return: Preprocessed corp
    """

    # Remove punctuations and numbers
    corp = re.sub('[^a-zA-Z]', ' ', corp)

    # Single character removal
    corp = re.sub(r"\s+[a-zA-Z]\s+", ' ', corp)

    # Removing multiple spaces
    corp = re.sub(r'\s+', ' ', corp)

    return corp


def get_pages(url_list):
    """
    Uses the requests module to scrape a list of urls for content.
    :param url_list: iterable of urls.
    :return: list of content strings corresponding to the input urls.
    """
    pagelist = []
    for url in url_list:
        response = requests.get(url, headers=headers)
        h = html2text.HTML2Text()
        h.ignore_links = True
        text = h.handle(response.text)
        pagelist.append(text)
    return pagelist


def create_xy_pairs(pages: list):
    """
    Creates feature - target pairs from the review text.
    :param pages: url contents, html tags removed
    :return: targets: the score assigned by the review to each card, corpi: the text of each review.
    """
    targets, corpi = [], []
    for page in pages:
        text = page.split('**Constructed:**')

        for sample in text[:-2]:
            try:
                targets.append(float(sample[:5]))
                corpi.append(preprocess_text(sample[5:]))
            except ValueError:
                pass

    return targets, corpi


if __name__ == "__main__":
    GET_REVIEWS = False

    if GET_REVIEWS:
        set_review_urls = set(get_review_urls())
        print(f"Scraping {len(set_review_urls)} unique Set Review pages.")
        pages = get_pages(set_review_urls)
        targs, corps = create_xy_pairs(pages)
        np.save(arr=np.array(targs), file="data/constructed_rating", allow_pickle=True)
        np.save(arr=np.array(corps), file="data/constructed_review", allow_pickle=True)
    else:
        targs = np.load("data/constructed_rating.npy")
        corps = np.load("data/constructed_review.npy")

    X = [corp.split("crystalcommerce")[0].split('png')[0] for corp in corps]
    lens = []
    for i, review in enumerate(X):
        doc = nlp(review.lower())
        tokens = [word.lemma_ for word in doc if not word in all_stopwords][:-3]
        X[i] = " ".join(tokens)
        lens.append(len(tokens))

    plt.hist(bins=10, x=lens)
    plt.show()

    y = np.array(list(1 if trg >= 2.5 else 0 for trg in targs))
    print(np.shape(X), np.shape(y))

    X_train, X_test, y_train, y_test, rating_train, rating_test, corps_train, corps_test = train_test_split(X, y, targs,
                                                                                                            X,
                                                                                                            test_size=0.20)
    tokenizer = Tokenizer(num_words=300)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 300

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    embeddings_dictionary = dict()
    glove_file = open(r"C:\Users\Joe\PycharmProjects\MTGSentimentAnalysis\wireframe\data\glove.6B\glove.6B.300d.txt",
                      encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions

    glove_file.close()

    embedding_matrix = np.zeros((vocab_size, 300))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    model = Sequential()
    embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(embedding_layer)

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=1e-6, l2=1e-6)))
    model.add(Dropout(rate=0.2))

    opt = keras.optimizers.Adam(learning_rate=1e-5)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

    print(model.summary())
    history = model.fit(X_train, y_train, batch_size=128, epochs=200, verbose=1)

    score = model.evaluate(X_test, y_test, verbose=1, return_dict=True)
    print(score.keys())

    plt.plot(history.history['loss'], label="loss")

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()

    y_pred = model.predict_proba(X_test)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred, pos_label=1)
    plt.plot(fpr, tpr)

    plt.show()

    correct = 0
    for i, (xt, yt, ytpred, rt, ct) in enumerate(zip(X_test, y_test, y_pred, rating_test, corps_test)):
        print(f"Pred: {ytpred} Actual: {rt} Mapped: {yt}")
        print(f"Corpus: {ct}")
        if yt == round(ytpred[0]):
            correct += 1

    print("Test Loss:", score['loss'])
    print("Test Accuracy:", score['acc'])

    print(f"AUC: {sklearn.metrics.auc(fpr, tpr)}")
    print(f"TEST ACCURACY: {correct / len(y_test)}")
