# DEFAULT LIBRARIES
import json
import pandas as pd
import re
import sys
import os
import numpy as np
import time
from operator import itemgetter
import gensim
from gensim.models import Word2Vec
from collections import defaultdict

# AUXILIAR LIBRARIES 
from utils import *
from text_preprocess import *

# MACHINE LEARNING LIBRARIES
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- CONSTANTS --- #
# from constants import *


if __name__ == "__main__":

    start_time = time.time()
    assert(len(sys.argv) >= 2)
    filename = sys.argv[1]

    yelp_data = False
    
    if len(sys.argv) > 2 and sys.argv[2] == 'yelp':
        # YELP DATASET
        yelp_data = True
        save_folder = '../dataset/' + filename[:-5] + "_data_" + sys.argv[4]
    else:
        # AMAZON DATASET
        save_folder = '../dataset/' + filename[:-7]+"_data_" + sys.argv[4]

    print(f"SAVE FOLDER： {save_folder}")

    # IF DIR NOT EXISTIS, CREATE THEM
    if not os.path.exists(save_folder + '/train'):
        os.makedirs(save_folder + '/train')
    if not os.path.exists(save_folder + '/val'):
        os.makedirs(save_folder + '/val')
    if not os.path.exists(save_folder + '/test'):
        os.makedirs(save_folder + '/test')

    if len(PRE_W2V_BIN_PATH) == 0:
        print("Warning: the word embedding file is not provided, will be initialized randomly")

    file = open(filename, errors='ignore')

    print(f"{now()}: Step1: loading raw review datasets...")

    users_id = []
    items_id = []
    ratings = []
    reviews = []

    # LOADING DATASET FROM FILE LINE PER LINE
    if yelp_data:
        for line in file:
            value = line.split('\t')
            reviews.append(value[3])
            users_id.append(value[0])
            items_id.append(value[1])
            ratings.append(value[2])
    else:
        for line in file:
            js = json.loads(line)
            if str(js['reviewerID']) == 'unknown':
                print("unknown user id")
                continue
            if str(js['asin']) == 'unknown':
                print("unkown item id")
                continue
            try:
                reviews.append(js['reviewText'])
                users_id.append(str(js['reviewerID']))
                items_id.append(str(js['asin']))
                ratings.append(str(js['overall']))
            except:
                continue

    # DATASET LOADED AND CREATED AS PANDAS DATAFRAME
    data_frame = {'user_id': pd.Series(users_id), 'item_id': pd.Series(items_id),
                  'ratings': pd.Series(ratings), 'reviews': pd.Series(reviews)}
    data = pd.DataFrame(data_frame)
    del users_id, items_id, ratings, reviews # CLEANNING MEMORY TRASH ;)


    # SOME INFORMATIONS
    uidList, iidList = get_count(data, 'user_id'), get_count(data, 'item_id')

    userNum_all = len(uidList)
    itemNum_all = len(iidList)
    print("===============Start:all  rawData size======================")
    print(f"dataNum: {data.shape[0]}")
    print(f"userNum: {userNum_all}")
    print(f"itemNum: {itemNum_all}")
    print(f"data densiy: {data.shape[0]/float(userNum_all * itemNum_all):.4f}")
    print("===============End: rawData size========================")

    '''
        FIRST EXTRACTION: USER/ITEM HASH TO ID,
        NUMERIZE DATA: MAP IN DATAFRAME USER/ITEM TO ID
    '''
    user2id = dict((uid, i) for(i, uid) in enumerate(uidList))
    item2id = dict((iid, i) for(i, iid) in enumerate(iidList))
    data = numerize(data, user2id=user2id, item2id=item2id)

    '''
        SECOND EXTRACTION: SPLIT TRAIN, TEST AND VALIDATION,
    '''
    print(f"-"*60)
    print(f"{now()} Step2: split datsets into train/val/test, save into npy data")
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=1234)
    
    uids_train, iids_train = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
    userNum = len(uids_train)
    itemNum = len(iids_train)
    print("===============Start: no-preprocess: trainData size======================")
    print("dataNum: {}".format(data_train.shape[0]))
    print("userNum: {}".format(userNum))
    print("itemNum: {}".format(itemNum))
    print("===============End: no-preprocess: trainData size========================")
    
    # === LDA PROCESSS === #

    # reviews = list(data_train[['reviews']].values)
    # reviews_for_lda = []
    # for rev in reviews:
    #     reviews_for_lda.extend(rev)


    # vectorizer = CountVectorizer(max_df=MAX_DF, min_df=2, max_features=MAX_VOCAB, stop_words='english')
    # tf = vectorizer.fit_transform(reviews_for_lda)
    # tf_reviews = tf.toarray()
    
    # no_topics = 20

    # lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

    # ===================== #

    # CHECKUP IN MISS IDS AND DROPOUT
    uidMiss = []
    iidMiss = []
    if userNum != userNum_all or itemNum != itemNum_all:
        for uid in range(userNum_all):
            if uid not in uids_train:
                uidMiss.append(uid)
        for iid in range(itemNum_all):
            if iid not in iids_train:
                iidMiss.append(iid)
                
                
    uid_index = []
    for uid in uidMiss:
        index = data_test.index[data_test['user_id'] == uid].tolist()
        uid_index.extend(index)
    data_train = pd.concat([data_train, data_test.loc[uid_index]])

    iid_index = []
    for iid in iidMiss:
        index = data_test.index[data_test['item_id'] == iid].tolist()
        iid_index.extend(index)
    data_train = pd.concat([data_train, data_test.loc[iid_index]])

    all_index = list(set().union(uid_index, iid_index))
    data_test = data_test.drop(all_index)

    data_test, data_val = train_test_split(data_test, test_size=0.5, random_state=1234)
    uidList_train, iidList_train = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')
    userNum = len(uidList_train)
    itemNum = len(iidList_train)
    print("===============Start--process finished: trainData size======================")
    print("dataNum: {}".format(data_train.shape[0]))
    print("userNum (config): {}".format(userNum))
    print("itemNum (config): {}".format(itemNum))
    print("===============End-process finished: trainData size========================")
    
    # SAVE TRAIN, TEST AND VALIDATION DATAFRAMES 
    # data.to_csv(f"{save_folder}/data.csv", index=False)
    data_train.to_csv(f"{save_folder}/train/Train.csv", index=False, escapechar='\\')
    data_test.to_csv(f"{save_folder}/test/Test.csv", index=False, escapechar='\\')
    data_val.to_csv(f"{save_folder}/val/Val.csv", index=False, escapechar='\\')

    # CREATING RAW RATING MATRIX
    rating_matrix = np.zeros([userNum + 2, itemNum + 2], dtype=np.single)

    '''
        THIRD EXTRACTION: EXTRACT INTERACTION - (USER, ITEM, RATING)
    '''
    def extract(data_dict, rx=False):
        x = []
        y = []
        for i in data_dict.values:
            uid = i[0]
            iid = i[1]
            x.append([uid, iid])
            y.append(float(i[2]))
            if rx:
                rating_matrix[uid][iid] = float(i[2])

        return x, y

    x_train, y_train = extract(data_train, rx=True)
    x_val, y_val = extract(data_val)
    x_test, y_test = extract(data_test)
    
    
    np.save(f"{save_folder}/train/Train.npy", x_train)
    np.save(f"{save_folder}/train/Train_Score.npy", y_train)
    np.save(f"{save_folder}/val/Val.npy", x_val)
    np.save(f"{save_folder}/val/Val_Score.npy", y_val)
    np.save(f"{save_folder}/test/Test.npy", x_test)
    np.save(f"{save_folder}/test/Test_Score.npy", y_test)
    np.save(f"{save_folder}/train/Rating_Matrix.npy", rating_matrix)

    print(now())
    print(f"Train data size (config): {len(x_train)}")
    print(f"Val data size (config): {len(x_val)}")
    print(f"Test data size (config): {len(x_test)}")

    '''
        FOURTH EXTRACTION: CONSTRUCT VOCAB AND USER/ITEM REVIEWS
    '''
    print(f"-"*60)
    print(f"{now()} Step3: Construct the vocab and user/item reviews from training set.")

    user_reviews_dict = {}
    item_reviews_dict = {}
    user_iid_dict = {}
    item_uid_dict = {}
    user_len = defaultdict(int)
    item_len = defaultdict(int)

    for i in data_train.values:
        str_review = clean_str(str(i[3]).encode('ascii', 'ignore').decode('ascii'))

        if len(str_review.strip()) == 0:
            str_review = "<unk>"

        if i[0] in user_reviews_dict:
            user_reviews_dict[i[0]].append(str_review)
            user_iid_dict[i[0]].append(i[1])
        else:
            user_reviews_dict[i[0]] = [str_review]
            user_iid_dict[i[0]] = [i[1]]

        if i[1] in item_reviews_dict:
            item_reviews_dict[i[1]].append(str_review)
            item_uid_dict[i[1]].append(i[0])
        else:
            item_reviews_dict[i[1]] = [str_review]
            item_uid_dict[i[1]] = [i[0]]

    # BUILDING USER/ITEM DOCUMENTS FROM REVIEWS
    vocab, user_review2doc, item_review2doc, user_reviews_dict, item_reviews_dict, topics = build_doc(user_reviews_dict, item_reviews_dict)  

    word_index = {}
    word_index['<unk>'] = 0
    for i, w in enumerate(vocab.keys(), 1):
        word_index[w] = i

    print(f"The vocab size: {len(word_index)}")
    print(f"Average user document length: {sum([len(i) for i in user_review2doc])/len(user_review2doc)}")
    print(f"Average item document length: {sum([len(i) for i in item_review2doc])/len(item_review2doc)}")
    
    vocab_df = pd.DataFrame({
        "words": list(word_index.keys()),
        "id":list(word_index.values())
    }) 
    
    vocab_df.to_csv(f"{save_folder}/train/vocab.csv", index=False)

    print(now())

    u_minNum, u_maxNum, u_averageNum, u_maxSent, u_minSent, u_pReviewLen, u_pSentLen = countNum(user_reviews_dict)
    print(f"u_max_r:{u_pReviewLen}")
    
    i_minNum, i_maxNum, i_averageNum, i_maxSent, i_minSent, i_pReviewLen, i_pSentLen = countNum(item_reviews_dict)
    print(f"i_max_r:{i_pReviewLen}")
        
    print("r_max_len：{}".format(max(u_pSentLen, i_pSentLen)))

    maxSentLen = max(u_pSentLen, i_pSentLen)
    minSentlen = 1
    


    print(f"-"*60)
    print(f"{now()} Step4: padding all the text and id lists and save into npy.")

    # PADDING USER TEXTS
    userReview2Index = []
    userDoc2Index = []
    user_iid_list = []
    for i in range(userNum):
        count_user = 0
        dataList = []
        a_count = 0

        textList = user_reviews_dict[i]
        
        u_iids = user_iid_dict[i]
        u_reviewList = []

        user_iid_list.append(padding_ids(u_iids, u_pReviewLen, itemNum+1))
        doc2index = [word_index[w] for w in user_review2doc[i]]

        for text in textList:
            text2index = []
            wordTokens = text.strip().split()

            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]

            if len(text2index) < maxSentLen:
                text2index = text2index + [0] * (maxSentLen - len(text2index))
            else:
                text2index = text2index[:maxSentLen]

            u_reviewList.append(text2index)

        userReview2Index.append(padding_text(u_reviewList, u_pReviewLen))
        userDoc2Index.append(doc2index)
        
        
    userDoc2Index, userDocLen = padding_doc(userDoc2Index)
    print(f"user document length: {userDocLen}")

    # PADDING ITEM TEXTS
    itemReview2Index = []
    itemDoc2Index = []
    item_uid_list = []
    for i in range(itemNum):
        count_item = 0
        dataList = []
        textList = item_reviews_dict[i]
        i_uids = item_uid_dict[i]
        i_reviewList = [] 
        i_reviewLen = [] 
        item_uid_list.append(padding_ids(i_uids, i_pReviewLen, userNum+1))
        doc2index = [word_index[w] for w in item_review2doc[i]]

        for text in textList:
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < maxSentLen:
                text2index = text2index + [0] * (maxSentLen - len(text2index))
            else:
                text2index = text2index[:maxSentLen]
            if len(text2index) < maxSentLen:
                text2index = text2index + [0] * (maxSentLen - len(text2index))
            i_reviewList.append(text2index)
        itemReview2Index.append(padding_text(i_reviewList, i_pReviewLen))
        itemDoc2Index.append(doc2index)

    itemDoc2Index, itemDocLen = padding_doc(itemDoc2Index)
    print(f"item document length: {itemDocLen}")

    print("-"*60)
    print(f"{now()} start writing npy...")
    np.save(f"{save_folder}/train/userReview2Index.npy", userReview2Index)
    np.save(f"{save_folder}/train/user_item2id.npy", user_iid_list)
    np.save(f"{save_folder}/train/userDoc2Index.npy", userDoc2Index)

    np.save(f"{save_folder}/train/itemReview2Index.npy", itemReview2Index)
    np.save(f"{save_folder}/train/item_user2id.npy", item_uid_list)
    np.save(f"{save_folder}/train/itemDoc2Index.npy", itemDoc2Index)

    np.save(f"{save_folder}/Topic_Matrix.npy", topics)

    print(f"{now()} write finised")

    '''
        WORD VECTOR MAPPING: W2V, FAXTEXT, BERT...
    '''
    print("-"*60)
    print(f"{now()} Step5: start word embedding mapping...")

    vocab_item = sorted(word_index.items(), key=itemgetter(1))
    w2v = []
    out = 0

    # INITIALIZE WORD EMBEDDINGS WITH PRE TRAINED
    # INSERT IN KWARGS THE TYPE OF PRETRAINED EMBEDDINGS

    if len(sys.argv) >= 4 and sys.argv[3] != 'default':
        PRE_W2V_BIN_PATH = sys.argv[3]

    if PRE_W2V_BIN_PATH:
        if sys.argv[4] == "fasttext":
            pre_word2v = gensim.models.KeyedVectors.load_word2vec_format(PRE_W2V_BIN_PATH, limit=999999)
        else:
            pre_word2v = gensim.models.KeyedVectors.load_word2vec_format(PRE_W2V_BIN_PATH, binary=True)
    else:
        pre_word2v = {}

    for word, key in vocab_item:
        if word in pre_word2v:
            w2v.append(pre_word2v[word])
        else:
            out += 1
            w2v.append(np.random.uniform(-1.0, 1.0, (300,)))

    print("############################")
    print(f"out of vocab: {out}")
    print(f"w2v size: {len(w2v)}")
    print("############################")

    w2vArray = np.array(w2v)
    print(f"Vocab Size and Word Dim: {w2vArray.shape}")
    np.save(f"{save_folder}/train/w2v.npy", w2v)
    end_time = time.time()
    print(f"{now()} all steps finised, cost time: {end_time-start_time:.4f}s")
