from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import h5py
import scipy.io as sio
import math
torch.multiprocessing.set_sharing_strategy('file_system')


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels
    def __getitem__(self, index):
        img = self.images[index, :]
        text = self.texts[index, :]
        labels = self.labels[index, :]
        return img, text, labels
    def __len__(self):
        count = len(self.images)
        return count

def construct_missed_data(I_tr, T_tr, alpha=0.0, beta=0.5):
    number = I_tr.shape[0]
    dual_size = math.ceil(number * alpha)
    only_image_size = math.floor((number - dual_size) * beta)
    only_text_size = number - dual_size - only_image_size
    print('Dual size: %d, Only img size: %d, Only txt size: %d' % (dual_size, only_image_size, only_text_size))
    random_idx = np.random.permutation(number)
    dual_index = random_idx[:dual_size]
    only_image_index = random_idx[dual_size:dual_size+only_image_size]
    only_text_index = random_idx[dual_size+only_image_size:dual_size+only_image_size+only_text_size]
    dual_img = I_tr[dual_index, :]
    dual_txt = T_tr[dual_index, :]
    only_img = I_tr[only_image_index, :]
    only_txt = T_tr[only_text_index, :]
    _dict = {'dual_img': dual_img, 'dual_txt': dual_txt, 'o_img': only_img, 'o_txt': only_txt}
    return _dict

def get_loader_flickr(alpha_train=0.0, beta_train=0.5):
    path = './datasets/MIRFlickr/'
    # x: images   y:tags   l:labels
    train_set = sio.loadmat(path + 'mir_train.mat')
    train_x = np.array(train_set['I_tr'], dtype=np.float)
    train_y = np.array(train_set['T_tr'], dtype=np.float)
    test_set = sio.loadmat(path + 'mir_query.mat')
    query_x = np.array(test_set['I_te'], dtype=np.float)
    query_y = np.array(test_set['T_te'], dtype=np.float)
    query_l = np.array(test_set['L_te'], dtype=np.float)
    db_set = sio.loadmat(path + 'mir_database.mat')
    retrieval_x = np.array(db_set['I_db'], dtype=np.float)
    retrieval_y = np.array(db_set['T_db'], dtype=np.float)
    retrieval_l = np.array(db_set['L_db'], dtype=np.float)
    train_com_data = {'I_tr': train_x, 'T_tr': train_y}
    query_data = {'I_te': query_x, 'T_te': query_y, 'L_te':query_l}
    retrieval_data = {'I_re': retrieval_x, 'T_re': retrieval_y, 'L_re': retrieval_l}
    train_missed_data = construct_missed_data(train_x, train_y, alpha=alpha_train, beta=beta_train)
    return (train_com_data, train_missed_data, query_data, retrieval_data)


def get_loader_nus(alpha_train=0.0, beta_train=0.5):
    path = './datasets/NUS-WIDE/'
    # x: images   y:tags   l:labels
    train_set = sio.loadmat(path + 'nus_train.mat')
    train_x = np.array(train_set['I_tr'], dtype=np.float)
    train_y = np.array(train_set['T_tr'], dtype=np.float)
    test_set = sio.loadmat(path + 'nus_query.mat')
    query_x = np.array(test_set['I_te'], dtype=np.float)
    query_y = np.array(test_set['T_te'], dtype=np.float)
    query_l = np.array(test_set['L_te'], dtype=np.float)
    db_set = sio.loadmat(path + 'nus_database.mat')
    retrieval_x = np.array(db_set['I_db'], dtype=np.float)
    retrieval_y = np.array(db_set['T_db'], dtype=np.float)
    retrieval_l = np.array(db_set['L_db'], dtype=np.float)
    train_com_data = {'I_tr': train_x, 'T_tr': train_y}
    query_data = {'I_te': query_x, 'T_te': query_y, 'L_te': query_l}
    retrieval_data = {'I_re': retrieval_x, 'T_re': retrieval_y, 'L_re': retrieval_l}
    train_missed_data = construct_missed_data(train_x, train_y, alpha=alpha_train, beta=beta_train)
    return (train_com_data, train_missed_data, query_data, retrieval_data)

def get_loader_coco(alpha_train=0.0, beta_train=0.5):

    path = './datasets/MSCOCO/'
    # x: images   y:tags   l:labels
    train_set = sio.loadmat(path + 'COCO_train.mat')
    train_x = np.array(train_set['I_tr'], dtype=np.float)
    train_y = np.array(train_set['T_tr'], dtype=np.float)
    test_set = sio.loadmat(path + 'COCO_query.mat')
    query_x = np.array(test_set['I_te'], dtype=np.float)
    query_y = np.array(test_set['T_te'], dtype=np.float)
    query_l = np.array(test_set['L_te'], dtype=np.float)
    db_set = h5py.File(path + 'COCO_database.mat', 'r', libver='latest', swmr=True)
    retrieval_l = np.array(db_set['L_db'], dtype=np.float).T
    retrieval_x = np.array(db_set['I_db'], dtype=np.float).T
    retrieval_y = np.array(db_set['T_db'], dtype=np.float).T
    db_set.close()
    train_com_data = {'I_tr': train_x, 'T_tr': train_y}
    query_data = {'I_te': query_x, 'T_te': query_y, 'L_te': query_l}
    retrieval_data = {'I_re': retrieval_x, 'T_re': retrieval_y, 'L_re': retrieval_l}
    train_missed_data = construct_missed_data(train_x, train_y, alpha=alpha_train, beta=beta_train)
    return (train_com_data, train_missed_data, query_data, retrieval_data)

