for dataset in Digital_Music_data Musical_Instruments_data Office_Products_data 
do
    for rep in default zeroshot finetunning
    do

        python3 main.py train --model=DeepCoNN --dataset=$dataset --emb_opt=word2vec --num-fea=1 --bert=$rep

        python3 main.py train --model=ConvMF --dataset=$dataset --emb_opt=word2vec --num-fea=1 --bert=$rep

        python3 main.py train --model=TARMF --dataset=$dataset --emb_opt=word2vec --num-fea=2 --bert=$rep

        python3 main.py train --model=NARRE --dataset=$dataset --emb_opt=word2vec --num-fea=2 --bert=$rep

        python3 main.py train --model=DAML --dataset=$dataset --emb_opt=word2vec --num-fea=2 --bert=$rep

    done

done