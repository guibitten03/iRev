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

        self.set_path(self.setting_path + self.emb_opt)

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