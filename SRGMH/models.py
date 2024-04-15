import torch
import torch.nn as nn
import torch.nn.functional as F


class encoder_template(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(encoder_template, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        out = self.fc1(x)
        out1 = self.bn(out)
        out2 = self.relu(out1)
        norm= self.norm(out)
        mu = self.relu(self.fc2(norm))
        logvar = self.relu(self.fc2(norm))
        return mu, logvar, out2


class decoder_template(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(decoder_template, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out1 = self.bn(out)
        out2 = self.relu(out1)
        norm = self.fc2(self.norm(out))
        out3 = self.relu(norm)
        return out3, out2


class MyNet(nn.Module):
    def __init__(self, code_len, ori_featI, ori_featT, latent_dim):
        super(MyNet, self).__init__()
        self.code_len = code_len
        self.encoderIMG = encoder_template(ori_featI, latent_dim)
        self.decoderIMG = decoder_template(latent_dim, ori_featI)
        self.encoderTXT = encoder_template(ori_featT, latent_dim)
        self.decoderTXT = decoder_template(latent_dim, ori_featT)
        self.FC = nn.Linear(latent_dim, latent_dim)
        self.BN = nn.BatchNorm1d(latent_dim)
        self.act = nn.ReLU(inplace=True)
        self.HJ = nn.Linear(latent_dim, code_len)
        self.img_fc = nn.Linear(2 * latent_dim, code_len)
        self.txt_fc = nn.Linear(2 * latent_dim, code_len)
        self.HC = nn.Linear(3 * code_len, code_len)
        self.HIBN = nn.BatchNorm1d(code_len)
        self.HTBN = nn.BatchNorm1d(code_len)
        self.HJBN = nn.BatchNorm1d(code_len)
        self.HBN = nn.BatchNorm1d(code_len)

    def forward(self, XI, XT, affinity_A):
        self.batch_num = XI.size(0)
        _, _, VI = self.encoderIMG(XI)
        VI = F.normalize(VI, dim=1)
        VI = self.FC(VI)
        S_VI = affinity_A.mm(VI)
        S_VI = self.act(self.BN(S_VI))
        _, _, VT = self.encoderTXT(XT)
        VT = F.normalize(VT, dim=1)
        VT = self.FC(VT)
        S_VT = affinity_A.mm(VT)
        S_VT = self.act(self.BN(S_VT))
        VC = torch.cat((VI, VT), 0)
        II = torch.eye(affinity_A.shape[0], affinity_A.shape[1]).cuda()
        S_cma = torch.cat((torch.cat((affinity_A, II), 1), torch.cat((II, affinity_A), 1)), 0)
        VJ = self.FC(VC)
        VJ = S_cma.mm(VJ)
        VJ = self.BN(VJ)
        VJ = VJ[:self.batch_num, :] + VJ[self.batch_num:, :]
        VJ_all = self.act(VJ)
        HJ = self.HJ(VJ)
        HJ = self.HJBN(HJ)
        HI = self.HIBN(self.img_fc(torch.cat((S_VI, VJ_all), 1)))
        HT = self.HTBN(self.txt_fc(torch.cat((VJ_all, S_VT), 1)))
        H = torch.tanh(self.HBN(self.HC(torch.cat((HI, HJ, HT), 1))))
        B = torch.sign(H)
        _, DeI_feat = self.decoderIMG(VI)
        _, DeT_feat = self.decoderTXT(VT)
        return HI, HT, H, B, DeI_feat, DeT_feat

class Img_Net(nn.Module):
    def __init__(self, code_len, img_feat_len):
        super(Img_Net, self).__init__()
        self.fc1 = nn.Linear(img_feat_len, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, code_len)
        self.tanh = nn.Tanh()
        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        HI = self.tanh(hid)
        return HI


class Txt_Net(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(Txt_Net, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, txt_feat_len)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(txt_feat_len, code_len)
        self.tanh = nn.Tanh()
        torch.nn.init.normal_(self.tohash.weight, mean=0.0, std=1)

    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        HT = self.tanh(hid)
        return HT