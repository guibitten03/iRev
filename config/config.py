# -*- coding: utf-8 -*-

# KEEP GLOBAL CONFIGURATION OF FRAMEWORK
# KEEP DATASET CONFIGURATIONS
import numpy as np

class DefaultConfig:
    model = ""
    dataset = ""

    # === BASE CONFIGURATION === #

    # PARALLELISM
    use_gpu = True
    gpu_id = 0
    multi_gpu = False
    gpu_ids = []
    seed = 2023
    num_workers = 16

    # TRAIN LOOP CONFIG
    num_epochs = 10
    batch_size = 128
    print_step = 100

    # OPTIMIZER CONFIG
    optimizer = 'Adam'
    weight_decay = 1e-3  # optimizer rameteri
    lr = 2e-3
    loss_method = 'mse'
    drop_out = 0.5

    # WORD AND TEXT EMBEDDINGS CONFIG
    use_word_embedding = True
    emb_opt = 'default' # Path to save train model print
    doc_len = 500
    word_dim = 300

    # LAYERS CONFIG #
    # --- CONVOLUTION --- #
    filters_num = 100
    kernel_size = 3

    # --- MLP --- #
    fc_dim = 32
    query_mlp_size = 128

    # ------------- #

    # FEATURES CONFIG
    id_emb_size = 32
    num_fea = 1  # id feature, review feature, doc feature
    use_review = True
    use_doc = True
    self_att = False

    # OUTPUT CONFIG
    r_id_merge = 'cat'  # review and ID feature
    ui_merge = 'cat'  # cat/add/dot
    output = 'lfm'  # 'fm', 'lfm', 'other: sum the ui_feature'
    direct_output = False

    # SAVE CONFIG
    fine_step = False  # save mode in step level, defualt in epoch
    pth_path = "checkpoints/MPCN_Video_Ga_data_default.pth"  # the saved pth path for test

    # TEST CONFIG
    ranking_metrics = False
    statistical_test = False

    # ALGORITHMS PARTICULAR CONFIGURATIONS
    hrdr = False
    topics = False
    man = False
    transnet = False


    def set_path(self, name):

        # DEFINE DATASET PATH
        self.data_root = f"./dataset/{name}"
        prefix = f"{self.data_root}/train"

        # DEFINE FEATURES PATH
        self.user_list_path = f'{prefix}/userReview2Index.npy'
        self.item_list_path = f'{prefix}/itemReview2Index.npy'

        self.user2itemid_path = f'{prefix}/user_item2id.npy'
        self.item2userid_path = f'{prefix}/item_user2id.npy'

        self.user_doc_path = f'{prefix}/userDoc2Index.npy'
        self.item_doc_path = f'{prefix}/itemDoc2Index.npy'

        self.ratingMatrix_path = f"{prefix}/Rating_Matrix.npy"

        self.topicMatrix_path = f"{self.data_root}/Topic_Matrix.npy"

        self.w2v_path = f'{prefix}/w2v.npy'

    def parse(self, kwargs):

        # EXCEPTION FOR NOT CORRESPONDING KEYS IN COMMAND
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        self.setting_path = self.setting_path + self.emb_opt
        self.set_path(self.setting_path)

        # LOAD DATASET FEATURES
        print("load npy from dist...")
        self.users_review_list = np.load(self.user_list_path, encoding='bytes')
        self.items_review_list = np.load(self.item_list_path, encoding='bytes')
        self.user2itemid_list = np.load(self.user2itemid_path, encoding='bytes')
        self.item2userid_list = np.load(self.item2userid_path, encoding='bytes')
        self.user_doc = np.load(self.user_doc_path, encoding='bytes')
        self.item_doc = np.load(self.item_doc_path, encoding='bytes')

        if self.topics:
            self.topic_matrix = np.load(self.topicMatrix_path, encoding='bytes')

        # OPTIONS CONFIGURATION
        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


class AMAZON_FASHION_data_Config(DefaultConfig):
    
    # DATASET FEATURES CONFIG
    setting_path = '.data/AMAZON_FASHION_data_'
        
    vocab_size = 1451
    
    r_max_len = 42

    u_max_r = 7
    i_max_r = 295

    train_data_size = 2529
    test_data_size = 315
    val_data_size = 316

    user_num = 404 + 2
    item_num = 31 + 2


class Toys_and_Games_data_Config(DefaultConfig):
    
    # DATASET FEATURES CONFIG
    setting_path = '.data/Toys_and_Games_data_'
        
    vocab_size = 50002
    
    r_max_len = 58

    u_max_r = 10
    i_max_r = 28

    train_data_size = 1462394
    test_data_size = 182701
    val_data_size = 182701

    user_num = 208143 + 2
    item_num = 78772 + 2

class Digital_Music_data_Config(DefaultConfig):
    
    # DATASET FEATURES CONFIG
    setting_path = '.data/Digital_Music_data_'
        
    vocab_size = 50002
    
    r_max_len = 34

    u_max_r = 12
    i_max_r = 18

    train_data_size = 135740
    test_data_size = 16942
    val_data_size = 16941

    user_num = 16561 + 2
    item_num = 11797 + 2


class Video_Games_data_Config(DefaultConfig):
    
    # DATASET FEATURES CONFIG
    setting_path = '.data/Video_Games_data_'
        
    vocab_size = 50002
    
    r_max_len = 172

    u_max_r = 10
    i_max_r = 35

    train_data_size = 397985
    test_data_size = 49717
    val_data_size = 49717

    user_num = 55217 + 2
    item_num = 17408 + 2


class Industrial_and_Scientific_data_Config(DefaultConfig):
    
    # DATASET FEATURES CONFIG
    setting_path = '.data/Industrial_and_Scientific_data_'
        
    vocab_size = 35956
    
    r_max_len = 62

    u_max_r = 7
    i_max_r = 17

    train_data_size = 61659
    test_data_size = 7701
    val_data_size = 7700

    user_num = 11041 + 2
    item_num = 5334 + 2

class Musical_Instruments_data_Config(DefaultConfig):
    
    # DATASET FEATURES CONFIG
    setting_path = '.data/Musical_Instruments_data_'
        
    vocab_size = 50002
    
    r_max_len = 75

    u_max_r = 9
    i_max_r = 24

    train_data_size = 185121
    test_data_size = 23112
    val_data_size = 23111

    user_num = 27528 + 2
    item_num = 10620 + 2


class Prime_Pantry_data_Config(DefaultConfig):
    
    # DATASET FEATURES CONFIG
    setting_path = '.data/Prime_Pantry_data_'
        
    vocab_size = 26453
    
    r_max_len = 35

    u_max_r =11
    i_max_r = 36

    train_data_size = 110090
    test_data_size = 13761
    val_data_size = 13760

    user_num = 14175 + 2
    item_num = 4970 + 2

class Office_Products_data_Config(DefaultConfig):
    
    # DATASET FEATURES CONFIG
    setting_path = '.data/Office_Products_data_'
        
    vocab_size = 50002
    
    r_max_len = 63

    u_max_r =9
    i_max_r = 31

    train_data_size = 640220
    test_data_size = 79962
    val_data_size = 79962

    user_num = 101498 + 2
    item_num = 27965 + 2

class Tamp_data_Config(DefaultConfig):
    
    # DATASET FEATURES CONFIG
    setting_path = '.data/Tamp_data_'
        
    vocab_size = 50002
    
    r_max_len = 91

    u_max_r = 3
    i_max_r = 64

    train_data_size = 387481
    test_data_size = 33704
    val_data_size = 33704

    user_num = 165718 + 2
    item_num = 9050 + 2


class Tucso_data_Config(DefaultConfig):
    
    # DATASET FEATURES CONFIG
    setting_path = '.data/Tucso_data_'
        
    vocab_size = 50002
    
    r_max_len = 93

    u_max_r = 3
    i_max_r = 56

    train_data_size = 339986
    test_data_size = 32447
    val_data_size = 32447

    user_num = 121312 + 2
    item_num = 9250 + 2


class Philladelphi_data_Config(DefaultConfig):
    
    # DATASET FEATURES CONFIG
    setting_path = '.data/Philladelphi_data_'
        
    vocab_size = 50002
    
    r_max_len = 86

    u_max_r = 3
    i_max_r = 84

    train_data_size = 811640
    test_data_size = 77956
    val_data_size = 77956

    user_num = 279857 + 2
    item_num = 14569 + 2