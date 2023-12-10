import time

from constants import *

# @GET NOEW TIME
def now(): 
    return str(time.strftime('%Y-%m-%d %H:%M:%S')) 

# @GET COUNT FROM IDS
def get_count(data, id):
    ids = set(data[id].tolist())
    return ids

# @NUMERIZE ID HASH TO INTEGER
def numerize(data, user2id, item2id):
    uid = list(map(lambda x: user2id[x], data['user_id']))
    iid = list(map(lambda x: item2id[x], data['item_id']))
    data['user_id'] = uid
    data['item_id'] = iid
    return data

# @PADDING TEXTS
def padding_text(textList, num):
    new_textList = []
    if len(textList) >= num:
        new_textList = textList[:num]
    else:
        padding = [[0] * len(textList[0]) for _ in range(num - len(textList))]
        new_textList = textList + padding
    return new_textList

def padding_ids(iids, num, pad_id):
    if len(iids) >= num:
        new_iids = iids[:num]
    else:
        new_iids = iids + [pad_id] * (num - len(iids))
            
    return new_iids

def padding_doc(doc):
        pDocLen = DOC_LEN
        new_doc = []
        for d in doc:
            if len(d) < pDocLen:
                d = d + [0] * (pDocLen - len(d))
            else:
                d = d[:pDocLen]
            new_doc.append(d)

        return new_doc, pDocLen
