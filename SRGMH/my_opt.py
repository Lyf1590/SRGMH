import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import *
import torch.utils.data as data
from torch.autograd import Variable
from models import MyNet, Img_Net, Txt_Net, decoder_template, encoder_template
from utils import compress, calculate_top_map, logger, p_topK, p_topK2
import numpy as np
import os.path as osp
import math


class SRGMH:
    def __init__(self, log, config):
        self.config = config
        self.log = log

        if config.dataset == 'MIRFlickr':
            complete_data, train_data, query_data, retrieval_data= get_loader_flickr(self.config.alpha_train, self.config.beta_train)
        elif config.dataset == 'NUS-WIDE':
            complete_data, train_data, query_data, retrieval_data = get_loader_nus(self.config.alpha_train, self.config.beta_train)
        elif config.dataset == 'COCO':
            complete_data, train_data, query_data, retrieval_data = get_loader_coco(self.config.alpha_train, self.config.beta_train)
        self.I_all_tr = complete_data['I_tr']
        self.T_all_tr = complete_data['T_tr']
        self.I_tr = train_data['dual_img']
        self.T_tr = train_data['dual_txt']
        self.only_I_tr = train_data['o_img']
        self.only_T_tr = train_data['o_txt']
        self.query_data = [query_data['I_te'], query_data['T_te'], query_data['L_te']]
        self.retrieval_data = [retrieval_data['I_re'], retrieval_data['T_re'], retrieval_data['L_re']]
        if self.I_tr.shape[0] == self.I_all_tr.shape[0]:
            self.log.info('**********No missing data samples!**********')
        else:
            self.log.info('**********There are missing data samples!**********')
        self.img_feat_len = self.I_tr.shape[1]
        self.txt_feat_len = self.T_tr.shape[1]
        self.dual_len = self.I_tr.shape[0]
        self.miss_img_len = self.only_I_tr.shape[0]
        self.miss_txt_len = self.only_T_tr.shape[0]
        self.train_num = self.dual_len + self.miss_txt_len + self.miss_img_len

        self.test_loader = data.DataLoader(CustomDataSet(self.query_data[0], self.query_data[1], self.query_data[2]), batch_size=self.config.batch_size, shuffle=False, num_workers=2)
        self.database_loader = data.DataLoader(CustomDataSet(self.retrieval_data[0], self.retrieval_data[1], self.retrieval_data[2]), batch_size=self.config.batch_size, shuffle=False, num_workers=2)
        self.batch_dual_size = math.ceil(self.config.batch_size * self.config.alpha_train)
        self.batch_img_size = math.floor((self.config.batch_size - self.batch_dual_size) * self.config.beta_train)
        self.batch_txt_size = math.floor((self.config.batch_size - self.batch_dual_size) * (1 - self.config.beta_train))
        self.batch_count = int(math.ceil(self.train_num / self.config.batch_size))
        self.encoder = {}
        self.decoder = {}
        self.reparameterize_with_noise = True
        self.hidden_size_rule = {'img': (2048, 1024),
                                 'txt': (1000, 700)}
        self.hidden_feat = [1024, 512]

        # Initialize model
        self.encoder['img'] = encoder_template(self.img_feat_len, self.config.latent_dim).cuda()
        self.encoder['txt'] = encoder_template(self.txt_feat_len, self.config.latent_dim).cuda()
        self.decoder['img'] = decoder_template(self.config.latent_dim, self.img_feat_len).cuda()
        self.decoder['txt'] = decoder_template(self.config.latent_dim, self.txt_feat_len).cuda()
        self.mynet = MyNet(code_len=self.config.hash_bit, ori_featI=self.img_feat_len, ori_featT=self.txt_feat_len, latent_dim=self.config.latent_dim).cuda()
        self.imgnet = Img_Net(code_len=self.config.hash_bit, img_feat_len=self.img_feat_len).cuda()
        self.txtnet = Txt_Net(code_len=self.config.hash_bit, txt_feat_len=self.txt_feat_len).cuda()
        self.opt_mynet = torch.optim.Adam(self.mynet.parameters(), lr=config.LR_MyNet)
        self.opt_imgnet = torch.optim.Adam(self.imgnet.parameters(), lr=config.LR_IMG)
        self.opt_txtnet = torch.optim.Adam(self.txtnet.parameters(), lr=config.LR_TXT)
        self.opt_en_img = torch.optim.SGD(self.encoder['img'].parameters(), lr=0.002, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
        self.opt_en_txt = torch.optim.SGD(self.encoder['txt'].parameters(), lr=0.002, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
        self.opt_de_img = torch.optim.SGD(self.decoder['img'].parameters(), lr=5e-3, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
        self.opt_de_txt = torch.optim.SGD(self.decoder['img'].parameters(), lr=5e-3, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
        self.record_Lsmodel = []
        self.record_Lshfunc = []

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size()).cuda()
            return mu + sigma * eps
        else:
            return mu

    def EntropicConfusion(self, z):
        softmax_out = nn.Softmax(dim=1)(z)
        batch_size = z.size(0)
        loss = torch.mul(softmax_out, torch.log(softmax_out)).sum() * ((1.0 / batch_size) + float("1e-8"))
        return loss

    def align_loss(self, ZI, ZT, S_batch):
        ZI_norm = F.normalize(ZI)
        ZT_norm = F.normalize(ZT)
        ZI_ZI = ZI_norm.mm(ZI_norm.t())
        ZT_ZT = ZT_norm.mm(ZT_norm.t())
        loss = self.config.lambda2 * (F.mse_loss(S_batch, ZI_ZI) + F.mse_loss(S_batch, ZT_ZT) + F.mse_loss(ZI_ZI, ZT_ZT))
        return loss

    def com_step(self, train_dual_img, train_dual_txt, train_only_img, train_only_txt):
        self.opt_en_img.zero_grad()
        self.opt_en_txt.zero_grad()
        self.opt_de_img.zero_grad()
        self.opt_de_txt.zero_grad()
        # only img
        mu_only_img, logvar_only_img, _ = self.encoder['img'](train_only_img)
        z_from_only_img = self.reparameterize(mu_only_img, logvar_only_img)
        only_img_from_img, _ = self.decoder['img'](z_from_only_img)
        # only txt
        mu_only_txt, logvar_only_txt, _ = self.encoder['txt'](train_only_txt)
        z_from_only_txt = self.reparameterize(mu_only_txt, logvar_only_txt)
        only_txt_from_txt, _ = self.decoder['txt'](z_from_only_txt)
        # single loss compute
        loss_single_img_recon = F.mse_loss(only_img_from_img, train_only_img)
        loss_single_txt_recon = F.mse_loss(only_txt_from_txt, train_only_txt)
        KLD = (((0.5 * torch.sum(1 + logvar_only_img - mu_only_img.pow(2) - logvar_only_img.exp())) / mu_only_img.size(0))
               + ((0.5 * torch.sum(1 + logvar_only_txt - mu_only_txt.pow(2) - logvar_only_txt.exp())) / mu_only_txt.size(0))).cuda()
        vaeloss = self.config.alpha1 * (loss_single_img_recon + loss_single_txt_recon) - 0.001 * KLD
        S_cmm = self.cal_similarity(z_from_only_img, z_from_only_txt, self.config.K1)
        align_loss = self.align_loss(z_from_only_img, z_from_only_txt, S_cmm)
        max_entropy_loss = self.config.alpha2 * align_loss
        # Generation of missing data
        txt_from_z_from_img, _ = self.decoder['txt'](z_from_only_img)
        mu_1, logvar_1, _ = self.encoder['txt'](txt_from_z_from_img)
        z_from_txt_from_z_from_img = self.reparameterize(mu_1, logvar_1)
        img_from_z_from_txt, _ = self.decoder['img'](z_from_only_txt)
        mu_2, logvar_2, _ = self.encoder['img'](img_from_z_from_txt)
        z_from_img_from_z_from_txt = self.reparameterize(mu_2, logvar_2)
        loss_new_recon = (F.mse_loss(z_from_only_img, z_from_txt_from_z_from_img) \
                 + F.mse_loss(z_from_only_txt, z_from_img_from_z_from_txt))
        loss_new_recon3 = self.config.alpha3 * loss_new_recon
        loss = vaeloss + max_entropy_loss + loss_new_recon3
        loss.backward()

        self.opt_en_img.step()
        self.opt_en_txt.step()
        self.opt_de_img.step()
        self.opt_de_txt.step()

        return loss.item(), img_from_z_from_txt, txt_from_z_from_img

    def train_com(self):
        self.encoder['img'].train()
        self.encoder['txt'].train()
        self.decoder['img'].train()
        self.decoder['txt'].train()
        Ls_method1 = 0
        dual_idx = np.arange(self.dual_len)
        img_idx = np.arange(self.miss_txt_len)
        txt_idx = np.arange(self.miss_img_len)
        np.random.shuffle(dual_idx)
        np.random.shuffle(img_idx)
        np.random.shuffle(txt_idx)
        if self.only_I_tr.shape[0] == 0 and self.only_T_tr.shape[0] == 0:
            I_com_tr = self.I_tr
            T_com_tr = self.T_tr
        else:
            for epoch in range(self.config.epoch):
                I_miss_r = list([])
                T_miss_r = list([])
                np.random.shuffle(dual_idx)
                np.random.shuffle(img_idx)
                np.random.shuffle(txt_idx)
                for batch_idx in range(self.batch_count):
                    small_idx_dual = dual_idx[batch_idx * self.batch_dual_size: (batch_idx + 1) * self.batch_dual_size]
                    small_idx_img = img_idx[batch_idx * self.batch_img_size: (batch_idx + 1) * self.batch_img_size]
                    small_idx_txt = txt_idx[batch_idx * self.batch_txt_size: (batch_idx + 1) * self.batch_txt_size]
                    train_dual_img = torch.FloatTensor(self.I_tr[small_idx_dual, :]).cuda()
                    train_dual_txt = torch.FloatTensor(self.T_tr[small_idx_dual, :]).cuda()
                    train_only_img = torch.FloatTensor(self.only_I_tr[small_idx_img, :]).cuda()
                    train_only_txt = torch.FloatTensor(self.only_T_tr[small_idx_txt, :]).cuda()
                    result = self.com_step(train_dual_img, train_dual_txt, train_only_img, train_only_txt)
                    loss, I_miss_com, T_miss_com = result
                    I_miss_r.append(I_miss_com)
                    T_miss_r.append(T_miss_com)
                    Ls_method1 = Ls_method1 + loss
                    if (batch_idx + 1) == self.batch_count:
                        self.log.info('[%4d/%4d] Loss: %.4f' % (epoch + 1, self.config.epoch, loss))

            I_miss_com_r = (torch.cat(I_miss_r, dim=0)).cpu().detach().numpy()
            T_miss_com_r = (torch.cat(T_miss_r, dim=0)).cpu().detach().numpy()
            I_com_tr = np.concatenate((self.I_tr, self.only_I_tr, I_miss_com_r), axis=0)
            T_com_tr = np.concatenate((self.T_tr, T_miss_com_r, self.only_T_tr), axis=0)
            self.record_Lsmodel.append(Ls_method1)

        return I_com_tr, T_com_tr

    def train_method(self, img, txt, epoch):
        coll_B = list([])
        Ls_method2 = 0
        img = Variable(torch.FloatTensor(img).cuda())
        txt = Variable(torch.FloatTensor(txt).cuda())
        S_cmm = self.cal_similarity(img, txt, self.config.K)
        self.mynet.train()
        for batch_idx in range(self.batch_count):
            if (self.config.batch_size * (batch_idx + 1) < self.train_num):
                S_batch = S_cmm[self.config.batch_size * batch_idx: self.config.batch_size * (batch_idx + 1), self.config.batch_size * batch_idx: self.config.batch_size * (batch_idx + 1)].cuda()
                Iind1, Iind2 = self.NN_select(S_batch)
                Tind1, Tind2 = self.NN_select(S_batch)
                imgNN1, imgNN2 = img[Iind1, :], img[Iind2, :]
                txtNN1, txtNN2 = txt[Tind1, :], txt[Tind2, :]
                img1 = img[self.config.batch_size * batch_idx: self.config.batch_size * (batch_idx + 1), :]
                txt1 = txt[self.config.batch_size * batch_idx: self.config.batch_size * (batch_idx + 1), :]
            else:
                S_batch = S_cmm[self.config.batch_size * batch_idx: , self.config.batch_size * batch_idx:].cuda()
                Iind1, Iind2 = self.NN_select(S_batch)
                Tind1, Tind2 = self.NN_select(S_batch)
                imgNN1, imgNN2 = img[Iind1, :], img[Iind2, :]
                txtNN1, txtNN2 = txt[Tind1, :], txt[Tind2, :]
                img1 = img[self.config.batch_size * batch_idx: , : ]
                txt1 = txt[self.config.batch_size * batch_idx: , : ]
            imgN = 0.7 * img1 + 0.2 * imgNN2 + 0.1 * imgNN1
            txtN = 0.7 * txt1 + 0.2 * txtNN2 + 0.1 * txtNN1
            HI, HT, H, B, DeI_feat, DeT_feat = self.mynet(imgN, txtN, S_batch)
            coll_B.extend(B.cpu().data.numpy())
            self.opt_mynet.zero_grad()
            loss_CCR = self.config.lambda1 * (F.mse_loss(DeI_feat, imgN) + F.mse_loss(DeT_feat, txtN))
            loss_2 = self.loss_method(HI, HT, H, B, S_batch)
            loss_model = loss_CCR + loss_2
            Ls_method2 = (Ls_method2 + loss_2).item()
            loss_model.backward()
            self.opt_mynet.step()
            if (batch_idx + 1) == self.batch_count:
                self.log.info('Epoch[%2d/%2d] Ls1=%.4f, Ls2=%.4f, Ls_model: %.4f' % (epoch + 1, self.config.epoch, loss_CCR, loss_2, loss_model.item()))

        coll_B = np.array(coll_B)
        self.record_Lsmodel.append(Ls_method2)
        return coll_B

    def train_Hashfunc(self, img_, txt_, coll_B, epoch):
        self.imgnet.train()
        self.txtnet.train()
        img_ = Variable(torch.FloatTensor(img_).cuda())
        txt_ = Variable(torch.FloatTensor(txt_).cuda())
        S_cmm = self.cal_similarity(img_, txt_, self.config.K)
        Ls_hfunc = 0
        B = torch.from_numpy(coll_B).cuda()
        img_norm = F.normalize(img_)
        txt_norm = F.normalize(txt_)
        num_cyc = img_norm.shape[0] / self.config.batch_size
        num_cyc = int(num_cyc+1) if num_cyc-int(num_cyc) > 0 else int(num_cyc)
        for kk in range(num_cyc):
            if kk+1 < num_cyc:
                img_batch = img_norm[kk * self.config.batch_size:(kk + 1) * self.config.batch_size, :]
                txt_batch = txt_norm[kk * self.config.batch_size:(kk + 1) * self.config.batch_size, :]
                B_batch = B[kk * self.config.batch_size:(kk + 1) * self.config.batch_size, :]
                S_batch = S_cmm[kk * self.config.batch_size:(kk + 1) * self.config.batch_size,
                                   kk * self.config.batch_size:(kk + 1) * self.config.batch_size]
            else:
                img_batch = img_norm[kk * self.config.batch_size:, :]
                txt_batch = txt_norm[kk * self.config.batch_size:, :]
                B_batch = B[kk * self.config.batch_size:, :]
                S_batch = S_cmm[kk * self.config.batch_size:, kk * self.config.batch_size:]
            hfunc_BI = self.imgnet(img_batch)
            hfunc_BT = self.txtnet(txt_batch)
            self.opt_imgnet.zero_grad()
            self.opt_txtnet.zero_grad()
            loss_f1 = F.mse_loss(hfunc_BI, B_batch) + F.mse_loss(hfunc_BT, B_batch) + F.mse_loss(hfunc_BI, hfunc_BT)
            S_BI_BT = F.normalize(hfunc_BI).mm(F.normalize(hfunc_BT).t())
            S_BI_BI = F.normalize(hfunc_BI).mm(F.normalize(hfunc_BI).t())
            S_BT_BT = F.normalize(hfunc_BT).mm(F.normalize(hfunc_BT).t())
            loss_f2 = F.mse_loss(S_BI_BT, S_batch) + F.mse_loss(S_BI_BI, S_batch) + F.mse_loss(S_BT_BT, S_batch)
            loss_hfunc = loss_f1 + self.config.beta * loss_f2
            Ls_hfunc = (Ls_hfunc + loss_hfunc).item()

            loss_hfunc.backward()
            self.opt_imgnet.step()
            self.opt_txtnet.step()

        self.log.info('Epoch [%2d/%2d], Ls_hfunc: %.4f' % (epoch + 1, self.config.epoch, loss_hfunc.item()))
        self.record_Lshfunc.append(Ls_hfunc)
        return self.imgnet, self.txtnet

    def performance_eval(self):
        self.log.info('--------------------Evaluation: mAP@50-------------------')
        self.imgnet.eval().cuda()
        self.txtnet.eval().cuda()
        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.imgnet, self.txtnet)
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        return MAP_I2T, MAP_T2I

    def eval2(self):
        self.log.info('--------------------Evaluation: mAP@50-------------------')
        self.imgnet.eval().cuda()
        self.txtnet.eval().cuda()
        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.imgnet, self.txtnet)
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        return MAP_I2T, MAP_T2I

    def NN_select(self, S):
        m, n1 = S.sort()
        ind1 = n1[:, -4:-1][:,0]
        ind2 = n1[:, -4:-1][:,1]
        return ind1, ind2

    def cal_similarity(self, F_I, F_T, K):
        batch_size = F_I.size(0)
        size = batch_size
        top_size = K
        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())
        S1 = self.config.a1 * S_I + (1 - self.config.a1) * S_T
        m, n1 = S1.sort()
        S1[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(-1)] = 0.
        S2 = torch.matmul(S_I, S_T.t())
        max_value = torch.max(S2)
        S2 = torch.exp((S2 - max_value) / self.config.temperature)
        S2 = S2 / torch.sum(S2, dim=1, keepdim=True)  # 归一化
        S = self.config.a2 * S1 + (1 - self.config.a2) * S2
        return S

    def loss_method(self, HI, HT, H, B, S_batch):
        HI_norm = F.normalize(HI)
        HT_norm = F.normalize(HT)
        H_norm = F.normalize(H)
        HI_HI = HI_norm.mm(HI_norm.t())
        HT_HT = HT_norm.mm(HT_norm.t())
        H_H = H_norm.mm(H_norm.t())
        I_T = HI_norm.mm(HT_norm.t())
        loss_1 = F.mse_loss(H_H, I_T) + F.mse_loss(HI_HI, HT_HT) + F.mse_loss(S_batch, H_H)
        loss = self.config.lambda2 * loss_1
        loss_DIS = F.mse_loss(H, B)
        return loss + loss_DIS

    def save_checkpoints(self):
        file_name = self.config.dataset + '_' + str(self.config.hash_bit) + 'bits.pth'
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.imgnet.state_dict(),
            'TxtNet': self.txtnet.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.log.info('**********Save the trained model successfully.**********')

    def load_checkpoints(self, file_name):
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.log.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.log.error('********** Fail to load checkpoint %s!*********' % ckp_path)
            raise IOError
        self.imgnet.load_state_dict(obj['ImgNet'])
        self.txtnet.load_state_dict(obj['TxtNet'])


