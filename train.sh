for dataset in Digital_Music_data Musical_Instruments_data Office_Products_data Tamp_data Tucso_data Philladelphi_data 
do

    # python3 main.py train --model=DeepCoNN --dataset=$dataset --emb_opt=word2vec --num-fea=1 

    # python3 main.py train --model=NARRE --dataset=$dataset --emb_opt=word2vec --num-fea=2 

    # python3 main.py train --model=MPCN --dataset=$dataset --emb_opt=word2vec --num-fea=1 

    # python3 main.py train --model=D_ATTN --dataset=$dataset --emb_opt=word2vec --num-fea=1 

    # python3 main.py train --model=DAML --dataset=$dataset --emb_opt=word2vec --num-fea=2 --batch_size=8 

    # python3 main.py train --model=ConvMF --dataset=$dataset --emb_opt=word2vec --num-fea=1 

    # python3 main.py train --model=TRANSNET --dataset=$dataset --emb_opt=word2vec --num-fea=1 --output=fm --transnet=True 

    # python3 main.py train --model=ANR --dataset=$dataset --emb_opt=word2vec --num-fea=1 --id_emb_size=500 

    # python3 main.py train --model=HRDR --dataset=$dataset --emb_opt=word2vec --num-fea=2 

    # python3 main.py train --model=TARMF --dataset=$dataset --emb_opt=word2vec --num-fea=2 

    # python3 main.py train --model=CARL --dataset=$dataset --num-fea=3 --emb_opt=word2vec 

    # python3 main.py train --model=ALFM --dataset=$dataset --num_fea=1 --topics=True --direct_output=True --emb_opt=word2vec 

    # python3 main.py train --model=A3NCF --dataset=$dataset --num_fea=1 --topics=True --direct_output=True --emb_opt=word2vec 

    # python3 main.py train --model=CARP --dataset=$dataset --output=lfm --emb_opt=word2vec 

    # python3 main.py train --model=CARM --dataset=$dataset --emb_opt=word2vec 

    python3 main.py train --model=MAN --dataset=$dataset --batch_size=32 --man=True --emb_opt=word2vec --output=nfm
done