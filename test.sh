for dataset in Tucso_data 
# for dataset in Digital_Music_data Musical_Instruments_data Office_Products_data Tamp_data Tucso_data Philladelphi_data 
do

    python3 main.py test --model=DeepCoNN --dataset=$dataset --emb_opt=word2vec --num-fea=1 --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=NARRE --dataset=$dataset --emb_opt=word2vec --num-fea=2 --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=MPCN --dataset=$dataset --emb_opt=word2vec --num-fea=1 --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=D_ATTN --dataset=$dataset --emb_opt=word2vec --num-fea=1 --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=DAML --dataset=$dataset --emb_opt=word2vec --num-fea=2 --batch_size=8 --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=ConvMF --dataset=$dataset --emb_opt=word2vec --num-fea=1 --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=TRANSNET --dataset=$dataset --emb_opt=word2vec --num-fea=1 --output=fm --transnet=True --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=ANR --dataset=$dataset --emb_opt=word2vec --num-fea=1 --id_emb_size=500 --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=HRDR --dataset=$dataset --emb_opt=word2vec --num-fea=2 --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=TARMF --dataset=$dataset --emb_opt=word2vec --num-fea=2 --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=CARL --dataset=$dataset --num-fea=3 --emb_opt=word2vec --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=ALFM --dataset=$dataset --num_fea=1 --topics=True --direct_output=True --emb_opt=word2vec --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=A3NCF --dataset=$dataset --num_fea=1 --topics=True --direct_output=True --emb_opt=word2vec --statistical_test=True --ranking_metrics=True

    python3 main.py test --model=CARP --dataset=$dataset --output=lfm --emb_opt=word2vec --statistical_test=True  --ranking_metrics=True

    python3 main.py test --model=CARM --dataset=$dataset --emb_opt=word2vec --statistical_test=True --ranking_metrics=True

    # python3 main.py test --model=MAN --dataset=$dataset --batch_size=16 --man=True --emb_opt=word2vec
done