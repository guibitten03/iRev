# python3 main.py train --model=DeepCoNN --dataset=Prime_Pantry_data --num-fea=1
python3 main.py train --model=DeepCoNN --dataset=Prime_Pantry_data --emb_opt=word2vec --num-fea=1
# python3 main.py train --model=DeepCoNN --dataset=Prime_Pantry_data --emb_opt=fasttext --num-fea=1

# python3 main.py train --model=NARRE --dataset=Prime_Pantry_data --num-fea=2
python3 main.py train --model=NARRE --dataset=Prime_Pantry_data --emb_opt=word2vec --num-fea=2
# python3 main.py train --model=NARRE --dataset=Prime_Pantry_data --emb_opt=fasttext --num-fea=2

# python3 main.py train --model=MPCN --dataset=Prime_Pantry_data --num-fea=1
python3 main.py train --model=MPCN --dataset=Prime_Pantry_data --emb_opt=word2vec --num-fea=1
# python3 main.py train --model=MPCN --dataset=Prime_Pantry_data --emb_opt=fasttext --num-fea=1

# python3 main.py train --model=D_ATTN --dataset=Prime_Pantry_data --num-fea=1
python3 main.py train --model=D_ATTN --dataset=Prime_Pantry_data --emb_opt=word2vec --num-fea=1
# python3 main.py train --model=D_ATTN --dataset=Prime_Pantry_data --emb_opt=fasttext --num-fea=1

# python3 main.py train --model=DAML --dataset=Prime_Pantry_data --num-fea=2 --batch_size=8
python3 main.py train --model=DAML --dataset=Prime_Pantry_data --emb_opt=word2vec --num-fea=2 --batch_size=8
# python3 main.py train --model=DAML --dataset=Prime_Pantry_data --emb_opt=fasttext --num-fea=2 --batch_size=8

# python3 main.py train --model=ConvMF --dataset=Prime_Pantry_data --num-fea=1
python3 main.py train --model=ConvMF --dataset=Prime_Pantry_data --emb_opt=word2vec --num-fea=1
# python3 main.py train --model=ConvMF --dataset=Prime_Pantry_data --emb_opt=fasttext --num-fea=1

# python3 main.py train --model=TRANSNET --dataset=Prime_Pantry_data --num-fea=1 --output=fm
# python3 main.py train --model=TRANSNET --dataset=Prime_Pantry_data --emb_opt=word2vec --num-fea=1 --output=fm
# python3 main.py train --model=TRANSNET --dataset=Prime_Pantry_data --emb_opt=fasttext --num-fea=1 --output=fm

# python3 main.py train --model=ANR --dataset=Prime_Pantry_data --num-fea=1 
python3 main.py train --model=ANR --dataset=Prime_Pantry_data --emb_opt=word2vec --num-fea=1 
# python3 main.py train --model=ANR --dataset=Prime_Pantry_data --emb_opt=fasttext --num-fea=1

# python3 main.py train --model=HRDR --dataset=Prime_Pantry_data --num-fea=2 
python3 main.py train --model=HRDR --dataset=Prime_Pantry_data --emb_opt=word2vec --num-fea=2 
# python3 main.py train --model=HRDR --dataset=Prime_Pantry_data --emb_opt=fasttext --num-fea=2 

python3 main.py train --model=TAERT --dataset=Prime_Pantry_data --num-fea=1 --emb_opt=word2vec

python3 main.py train --model=CARL --dataset=Prime_Pantry_data --num-fea=3 --emb_opt=word2vec

python3 main.py train --model=ALFM --dataset=Prime_Pantry_data --num_fea=1 --topics=True --direct_output=True --emb_opt=word2vec

python3 main.py train --model=A3NCF --dataset=Prime_Pantry_data --num_fea=1 --topics=True --direct_output=True --emb_opt=word2vec

python3 main.py train --model=CARP --dataset=Prime_Pantry_data --output=nfm --emb_opt=word2vec

# python3 main.py train --model=MAN --dataset=Prime_Pantry_data --batch_size=2 --man=True