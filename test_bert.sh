for dataset in Digital_Music_data Musical_Instruments_data Office_Products_data 
do
    for rep in default zeroshot finetunning
    do

        python3 main.py test --model=DeepCoNN --dataset=$dataset --emb_opt=word2vec --num-fea=1 --bert=$rep --statistical_test=True --ranking_metrics=True

        python3 main.py test --model=ConvMF --dataset=$dataset --emb_opt=word2vec --num-fea=1 --bert=$rep --statistical_test=True --ranking_metrics=True

        python3 main.py test --model=TARMF --dataset=$dataset --emb_opt=word2vec --num-fea=2 --bert=$rep --statistical_test=True --ranking_metrics=True

        python3 main.py test --model=NARRE --dataset=$dataset --emb_opt=word2vec --num-fea=2 --bert=$rep --statistical_test=True --ranking_metrics=True

        python3 main.py test --model=DAML --dataset=$dataset --emb_opt=word2vec --num-fea=2 --bert=$rep --statistical_test=True --ranking_metrics=True

    done

done