# python3 main.py test --model=DeepCoNN --dataset=AMAZON_FASHION_data --num-fea=1
# python3 main.py test --model=DeepCoNN --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=1
# python3 main.py test --model=DeepCoNN --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=1

# python3 main.py test --model=NARRE --dataset=AMAZON_FASHION_data --num-fea=2 --statistical_test=True
# python3 main.py test --model=NARRE --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=2 --statistical_test=True
# python3 main.py test --model=NARRE --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=2 --statistical_test=True

# python3 main.py test --model=MPCN --dataset=AMAZON_FASHION_data --num-fea=1 --statistical_test=True
# python3 main.py test --model=MPCN --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=1 --statistical_test=True
# python3 main.py test --model=MPCN --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=1 --statistical_test=True

# python3 main.py test --model=D_ATTN --dataset=AMAZON_FASHION_data --num-fea=1 --statistical_test=True
# python3 main.py test --model=D_ATTN --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=1 --statistical_test=True
# python3 main.py test --model=D_ATTN --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=1 --statistical_test=True

# python3 main.py test --model=DAML --dataset=AMAZON_FASHION_data --num-fea=2 --statistical_test=True --batch_size=8
# python3 main.py test --model=DAML --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=2 --statistical_test=True --batch_size=8
# python3 main.py test --model=DAML --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=2 --statistical_test=True --batch_size=8

# python3 main.py test --model=ConvMF --dataset=AMAZON_FASHION_data --num-fea=1 --statistical_test=True
# python3 main.py test --model=ConvMF --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=1 --statistical_test=True
# python3 main.py test --model=ConvMF --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=1 --statistical_test=True 

# python3 main.py test --model=TRANSNET --dataset=AMAZON_FASHION_data --num-fea=1 --output=fm --statistical_test=True --transnet=True
# python3 main.py test --model=TRANSNET --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=1 --output=fm --statistical_test=True --transnet=True
# python3 main.py test --model=TRANSNET --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=1 --output=fm --statistical_test=True --transnet=True

python3 main.py test --model=ANR --dataset=AMAZON_FASHION_data --num-fea=1 --direct_output=True --statistical_test=True
python3 main.py test --model=ANR --dataset=AMAZON_FASHION_data --emb_opt=word2vec --num-fea=1 --direct_output=True --statistical_test=True
python3 main.py test --model=ANR --dataset=AMAZON_FASHION_data --emb_opt=fasttext --num-fea=1 --direct_output=True --statistical_test=True

python3 main.py test --model=TAERT --dataset=AMAZON_FASHION_data --num-fea=1 --statistical_test=True

python3 main.py test --model=CARL --dataset=AMAZON_FASHION_data --num-fea=1 --statistical_test=True --batch_size=64

python3 main.py test --model=ALFM --dataset=AMAZON_FASHION_data --num_fea=1 --topics=True --direct_output=True --batch_size=4 --statistical_test=True

python3 main.py test --model=A3NCF --dataset=AMAZON_FASHION_data --num_fea=1 --topics=True --direct_output=True --batch_size=64 --statistical_test=True

python3 main.py test --model=CARP --dataset=AMAZON_FASHION_data --batch_size=8 --statistical_test=True --output=nfm

python3 main.py test --model=MAN --dataset=AMAZON_FASHION_data --direct_output=True --statistical_test=True