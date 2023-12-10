python3 pro_data.py .data/AMAZON_FASHION_5.json amazon default default > info/AMAZON_FASHION_INFO_default

python3 pro_data.py .data/AMAZON_FASHION_5.json amazon language_models/GoogleNews-vectors-negative300.bin word2vec > info/AMAZON_FASHION_INFO_w2v

python3 pro_data.py .data/AMAZON_FASHION_5.json amazon language_models/wiki-news-300d-1M.vec fasttext > info/AMAZON_FASHION_INFO_ft

python3 pro_data.py .data/Video_Games_5.json amazon default default > info/VIDEO_GAMES_INFO_default

python3 pro_data.py .data/Musical_Instruments_5.json amazon default default > info/MUSICAL_INSTRUMENT_INFO_default

python3 pro_data.py .data/Industrial_and_Scientific_5.json amazon default default > info/INDUSTRIAL_AND_SCIENTIFIC_INFO_default

python3 pro_data.py .data/All_Beauty_5.json amazon default default > info/ALL_BEAUTY_INFO_default