# python3 main.py train --model=DeepCoNN --dataset=AMAZON_FASHION_data --num-fea=1
python3 main.py train --model=DeepCoNN --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=1
# python3 main.py train --model=DeepCoNN --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=1

# python3 main.py train --model=NARRE --dataset=AMAZON_FASHION_data --num-fea=2
python3 main.py train --model=NARRE --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=2
# python3 main.py train --model=NARRE --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=2

# python3 main.py train --model=MPCN --dataset=AMAZON_FASHION_data --num-fea=1
python3 main.py train --model=MPCN --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=1
# python3 main.py train --model=MPCN --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=1

# python3 main.py train --model=D_ATTN --dataset=AMAZON_FASHION_data --num-fea=1
python3 main.py train --model=D_ATTN --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=1
# python3 main.py train --model=D_ATTN --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=1

# python3 main.py train --model=DAML --dataset=AMAZON_FASHION_data --num-fea=2 --batch_size=8
python3 main.py train --model=DAML --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=2 --batch_size=8
# python3 main.py train --model=DAML --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=2 --batch_size=8

# python3 main.py train --model=ConvMF --dataset=AMAZON_FASHION_data --num-fea=1
python3 main.py train --model=ConvMF --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=1
# python3 main.py train --model=ConvMF --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=1

# python3 main.py train --model=TRANSNET --dataset=AMAZON_FASHION_data --num-fea=1 --output=fm
# python3 main.py train --model=TRANSNET --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=1 --output=fm
# python3 main.py train --model=TRANSNET --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=1 --output=fm

# python3 main.py train --model=ANR --dataset=AMAZON_FASHION_data --num-fea=1 
python3 main.py train --model=ANR --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=1 
# python3 main.py train --model=ANR --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=1

# python3 main.py train --model=HRDR --dataset=AMAZON_FASHION_data --num-fea=2 
python3 main.py train --model=HRDR --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=2 
# python3 main.py train --model=HRDR --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=2 

python3 main.py train --model=TAERT --dataset=AMAZON_FASHION_data --num-fea=1 --emb_opt=word2vec

python3 main.py train --model=CARL --dataset=AMAZON_FASHION_data --num-fea=3 --emb_opt=word2vec

python3 main.py train --model=ALFM --dataset=AMAZON_FASHION_data --num_fea=1 --topics=True --direct_output=True --emb_opt=word2vec

python3 main.py train --model=A3NCF --dataset=AMAZON_FASHION_data --num_fea=1 --topics=True --direct_output=True --emb_opt=word2vec

python3 main.py train --model=CARP --dataset=AMAZON_FASHION_data --output=nfm --emb_opt=word2vec

# python3 main.py train --model=MAN --dataset=AMAZON_FASHION_data --batch_size=2 --man=True