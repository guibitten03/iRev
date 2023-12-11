# python3 main.py test --model=DeepCoNN --dataset=Industrial_and_Scientific_data --num-fea=1
python3 main.py test --model=DeepCoNN --dataset=Industrial_and_Scientific_data --emb_opt=word2vec --num-fea=1
# python3 main.py test --model=DeepCoNN --dataset=Industrial_and_Scientific_data --emb_opt=fasttext --num-fea=1

# python3 main.py test --model=NARRE --dataset=Industrial_and_Scientific_data --num-fea=2 
python3 main.py test --model=NARRE --dataset=Industrial_and_Scientific_data --emb_opt=word2vec --num-fea=2 
# python3 main.py test --model=NARRE --dataset=Industrial_and_Scientific_data --emb_opt=fasttext --num-fea=2 

# python3 main.py test --model=MPCN --dataset=Industrial_and_Scientific_data --num-fea=1 
python3 main.py test --model=MPCN --dataset=Industrial_and_Scientific_data --emb_opt=word2vec --num-fea=1 
# python3 main.py test --model=MPCN --dataset=Industrial_and_Scientific_data --emb_opt=fasttext --num-fea=1 

# python3 main.py test --model=D_ATTN --dataset=Industrial_and_Scientific_data --num-fea=1 
python3 main.py test --model=D_ATTN --dataset=Industrial_and_Scientific_data --emb_opt=word2vec --num-fea=1 
# python3 main.py test --model=D_ATTN --dataset=Industrial_and_Scientific_data --emb_opt=fasttext --num-fea=1 

# python3 main.py test --model=DAML --dataset=Industrial_and_Scientific_data --num-fea=2  --batch_size=8
python3 main.py test --model=DAML --dataset=Industrial_and_Scientific_data --emb_opt=word2vec --num-fea=2  --batch_size=8
# python3 main.py test --model=DAML --dataset=Industrial_and_Scientific_data --emb_opt=fasttext --num-fea=2  --batch_size=8

# python3 main.py test --model=ConvMF --dataset=Industrial_and_Scientific_data --num-fea=1 
python3 main.py test --model=ConvMF --dataset=Industrial_and_Scientific_data --emb_opt=word2vec --num-fea=1 
# python3 main.py test --model=ConvMF --dataset=Industrial_and_Scientific_data --emb_opt=fasttext --num-fea=1  

# python3 main.py test --model=TRANSNET --dataset=Industrial_and_Scientific_data --num-fea=1 --output=fm  --transnet=True
# python3 main.py test --model=TRANSNET --dataset=Industrial_and_Scientific_data --emb_opt=word2vec --num-fea=1 --output=fm  --transnet=True
# python3 main.py test --model=TRANSNET --dataset=Industrial_and_Scientific_data --emb_opt=fasttext --num-fea=1 --output=fm  --transnet=True

# python3 main.py test --model=ANR --dataset=Industrial_and_Scientific_data --num-fea=1 --direct_output=True 
python3 main.py test --model=ANR --dataset=Industrial_and_Scientific_data --emb_opt=word2vec --num-fea=1 
# python3 main.py test --model=ANR --dataset=Industrial_and_Scientific_data --emb_opt=fasttext --num-fea=1 --direct_output=True 

python3 main.py test --model=TAERT --dataset=Industrial_and_Scientific_data --num-fea=1 --emb_opt=word2vec

python3 main.py test --model=CARL --dataset=Industrial_and_Scientific_data --num-fea=3 --emb_opt=word2vec

python3 main.py test --model=ALFM --dataset=Industrial_and_Scientific_data --num_fea=1 --topics=True --direct_output=True --emb_opt=word2vec

python3 main.py test --model=A3NCF --dataset=Industrial_and_Scientific_data --num_fea=1 --topics=True --direct_output=True --emb_opt=word2vec

python3 main.py test --model=CARP --dataset=Industrial_and_Scientific_data  --output=nfm --emb_opt=word2vec

# python3 main.py test --model=MAN --dataset=Industrial_and_Scientific_data --direct_output=True 