from my_opt import *
import argparse
import torch
from utils import logger


def param_list(log, config):
    log.info('>>> Configs List <<<')
    log.info('--- Dadaset:{}'.format(config.dataset))
    log.info('--- NUM_EPOCH:{}'.format(config.epoch))
    log.info('--- alpha_train:{}'.format(config.alpha_train))
    log.info('--- beta_train:{}'.format(config.beta_train))
    log.info('--- SEED:{}'.format(config.seed))
    log.info('--- Bit:{}'.format(config.hash_bit))
    log.info('--- Batch:{}'.format(config.batch_size))
    log.info('--- Lr_IMG:{}'.format(config.LR_IMG))
    log.info('--- Lr_TXT:{}'.format(config.LR_TXT))
    log.info('--- LR_MyNet:{}'.format(config.LR_MyNet))
    log.info('--- lambda1:{}'.format(config.lambda1))
    log.info('--- lambda2:{}'.format(config.lambda2))
    log.info('--- beta:{}'.format(config.beta))
    log.info('--- K:{}'.format(config.K))
    log.info('--- K1:{}'.format(config.K1))
    log.info('--- temperature:{}'.format(config.temperature))
    log.info('--- a1:{}'.format(config.a1))
    log.info('--- a2:{}'.format(config.a2))
    log.info('--- alpha1:{}'.format(config.alpha1))
    log.info('--- alpha2:{}'.format(config.alpha2))
    log.info('--- alpha3:{}'.format(config.alpha3))


def main(config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    torch.cuda.set_device(config.gpu_ID)
    logName = config.dataset + '_' + str(config.hash_bit)
    log = logger(logName)
    param_list(log, config)
    model = SRGMH(log, config)
    best_it = best_ti = 0

    if config.train == True:
        print('Training stage!')
        I_com_r, T_com_r = model.train_com()
        for epoch in range(config.epoch):
            coll_B = model.train_method(I_com_r, T_com_r, epoch)
            model.train_Hashfunc(I_com_r, T_com_r, coll_B, epoch)
            if (epoch + 1) % config.EVAL_INTERVAL == 0:
                MAP_I2T, MAP_T2I = model.performance_eval()
                if (best_it + best_ti) < (MAP_I2T + MAP_T2I):
                    best_it, best_ti = MAP_I2T, MAP_T2I
                log.info('mAP@50 I->T: %.3f, mAP@50 T->I: %.3f' % (MAP_I2T, MAP_T2I))
                log.info('Best MAP of I->T: %.3f, Best mAP of T->I: %.3f' % (best_it, best_ti))
                log.info('--------------------------------------------------------------------')
            if epoch + 1 == config.epoch:
                model.save_checkpoints()
    else:
        ckp = config.dataset + '_' + str(config.hash_bit)+'bits.pth'
        model.load_checkpoints(ckp)
        MAP_I2T, MAP_T2I = model.performance_eval()
        log.info('mAP@50 I->T: %.3f, mAP@50 T->I: %.3f' % (MAP_I2T, MAP_T2I))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ours')
    parser.add_argument('--train', default=True, help='train or test', type=bool)
    parser.add_argument('--dataset', default='NUS-WIDE', help='MIRFlickr, NUS-WIDE or COCO', type=str)
    parser.add_argument('--alpha_train', type=float, default=1, help='Missing ratio of train set--image.')
    parser.add_argument('--beta_train', type=float, default=0, help='Missing ratio of query set--txt.')
    parser.add_argument('--latent_dim', default=512, type=int, help='512,1024')
    parser.add_argument('--lambda1', default=8, type=float, help='10')
    parser.add_argument('--lambda2', default=1, type=float, help='1')
    parser.add_argument('--beta', default=1, type=float, help='0.01')
    parser.add_argument('--alpha1', type=float, default=0.5)
    parser.add_argument('--alpha2', type=float, default=0.7)
    parser.add_argument('--alpha3', type=float, default=0.7)
    parser.add_argument('--LR_IMG', default=0.0001, type=float, help='0.0001')
    parser.add_argument('--LR_TXT', default=0.0001, type=float, help='0.0001')
    parser.add_argument('--LR_MyNet', default=0.0001, type=float, help='0.0001')
    parser.add_argument('--MOMENTUM', default=0.9, type=float, help='0.8')
    parser.add_argument('--WEIGHT_DECAY', default=5e-4, type=float, help='1e-4')
    parser.add_argument('--K', default=3300, help='3300', type=int)
    parser.add_argument('--K1', default=30, type=int, help='number of Missing ratio: 30、70、110、150')
    parser.add_argument('--temperature', default=3, help='1', type=int)
    parser.add_argument('--a1', default=0.6, help='balance ST and SI (0.5)', type=float)
    parser.add_argument('--a2', default=0.6, help='balance S1 and S2 (0.5)',type=float)
    parser.add_argument('--hash_bit', default=16, help='code length', type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--gpu_ID', default=1, type=int)
    parser.add_argument('--seed', default=3407, type=int)  # Please choose a suitable random seed.
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--EPOCH_INTERVAL', default=2, type=int)
    parser.add_argument('--epoch', default=120, type=int)
    parser.add_argument('--EVAL_INTERVAL', default=10, type=int)
    parser.add_argument('--Latent_Size', default=512, type=int)
    parser.add_argument('--MODEL_DIR', default="./checkpoints", type=str)

    config = parser.parse_args()
    main(config)

