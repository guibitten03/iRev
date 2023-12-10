import sys

from gensim.test.utils import common_texts
from gensim.models import Word2Vec

if __name__ == "__main__":

    assert(len(sys.argv) > 2)
    model_emb = sys.argv[1]
    vector_size = int(sys.argv[2])

    if model_emb == "word2vec":
        model = Word2Vec(sentences=common_texts,
                         vector_size=vector_size,
                         window=5,
                         min_count=1,
                         workers=4)
        model.save("language_models/word2vec.model")