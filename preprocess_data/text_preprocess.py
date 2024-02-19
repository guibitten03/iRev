import re
import numpy as np
import math

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


from constants import *


# @CLEAR STRING
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()


# @FUNCTION TO BUILD VOCABULARY
def bulid_vocbulary(xDict):
    rawReviews = []
    for (id, text) in xDict.items():
        rawReviews.append(' '.join(text))
    return rawReviews


# @FUNCTION TO BUILD DOCUMENT FROM REVIEWS
def build_doc(u_reviews_dict, i_reviews_dict):

    u_reviews = []
    for ind in range(len(u_reviews_dict)):
        u_reviews.append(' <SEP> '.join(u_reviews_dict[ind]))

    i_reviews = []
    for ind in range(len(i_reviews_dict)):
        i_reviews.append('<SEP>'.join(i_reviews_dict[ind]))

    # @TF-IDF VOCABULARY 
    vectorizer = TfidfVectorizer(max_df=MAX_DF, max_features=MAX_VOCAB)
    vectorizer.fit(u_reviews)

    tf = vectorizer.transform(u_reviews + i_reviews)

    rev = tf.toarray()

    # --- TOPIC MODELING --- #
    no_topics = 32
    lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    topics = lda.transform(rev)

    print(f"LDA transform matrix: {topics.shape}")
    

    vocab = vectorizer.vocabulary_
    vocab[MAX_VOCAB] = '<SEP>'

    def clean_review(rDict):
        new_dict = {}
        for k, text in rDict.items():
            new_reviews = []
            for r in text:
                words = ' '.join([w for w in r.split() if w in vocab])
                new_reviews.append(words)
            new_dict[k] = new_reviews
        return new_dict

    def clean_doc(raw):
        new_raw = []
        for line in raw:
            review = [word for word in line.split() if word in vocab]
            if len(review) > DOC_LEN:
                review = review[:DOC_LEN]
            new_raw.append(review)
        return new_raw

    u_reviews_dict = clean_review(u_reviews_dict)
    i_reviews_dict = clean_review(i_reviews_dict)

    u_doc = clean_doc(u_reviews)
    i_doc = clean_doc(i_reviews)

    return vocab, u_doc, i_doc, u_reviews_dict, i_reviews_dict, topics


def build_bert_doc(u_bert_dict, i_bert_dict):
    u_bert = []
    u_rev_mean = 0.0
    for ind in range(len(u_bert_dict)):
        revs = u_bert_dict[ind]
        u_rev_mean += len(revs)
        doc = []
        for d in revs:
            doc.extend(d)

        u_bert.append(doc)

    i_bert = []
    i_rev_mean = 0.0
    for ind in range(len(i_bert_dict)):
        revs = i_bert_dict[ind]
        i_rev_mean += len(revs)
        doc = []
        for d in revs:
            doc.extend(d)

        i_bert.append(doc)
    

    def clean_rev(rev_dict, rev_size):
        new_dict = []
        for k, v in rev_dict.items():
            if len(v) < rev_size:
                v_pad = []
                pad = rev_size - len(v)
                for i in range(pad):
                    padding = [0] * 768
                    v_pad.append(padding)
                v.extend(v_pad)
            
            new_dict.append(v[:rev_size])
        
        return new_dict
        
    def clean_doc(raw, doc_len):
        new_raw = []
        for line in raw:
            if len(line) < doc_len:
                line = np.pad(line, (doc_len - len(line)), "constant")
            if len(line) > doc_len:
                line = line[:doc_len]
            new_raw.append(line)
        return new_raw
    
    u_dict = clean_rev(u_bert_dict, math.floor((u_rev_mean / len(u_bert_dict))))
    i_dict = clean_rev(i_bert_dict, math.floor((i_rev_mean / len(i_bert_dict))))
    
    u_doc = clean_doc(u_bert, math.floor((u_rev_mean / len(u_bert_dict)) * 768))
    i_doc = clean_doc(i_bert, math.floor((i_rev_mean / len(i_bert_dict)) * 768))

    return u_dict, i_dict, u_doc, i_doc


# @DESCRIBE DATASET FOR CONFIGURATION
def countNum(xDict):
    
    minNum = 100
    maxNum = 0
    sumNum = 0
    maxSent = 0
    minSent = 3000
    ReviewLenList = []
    SentLenList = []

    for (i, text) in xDict.items():

        sumNum = sumNum + len(text)
        if len(text) < minNum:
            minNum = len(text)
        if len(text) > maxNum:
            maxNum = len(text)

        ReviewLenList.append(len(text))

        for sent in text:

            if sent != "":
                wordTokens = sent.split()
            if len(wordTokens) > maxSent:
                maxSent = len(wordTokens)
            if len(wordTokens) < minSent:
                minSent = len(wordTokens)

            SentLenList.append(len(wordTokens))
            
    averageNum = sumNum // (len(xDict))

    x = np.sort(SentLenList)
    xLen = len(x)
    pSentLen = x[int(P_REVIEW * xLen) - 1]
    x = np.sort(ReviewLenList)
    xLen = len(x)
    pReviewLen = x[int(P_REVIEW * xLen) - 1]

    return minNum, maxNum, averageNum, maxSent, minSent, pReviewLen, pSentLen