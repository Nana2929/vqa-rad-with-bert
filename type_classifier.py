# -*- coding: utf-8 -*-#
# [PubMedClip]
#-------------------------------------------------------------------------------
# Name:         classify_question
# Description:
# Author:       Boliu.Kelvin
# Date:         2020/5/14
#-------------------------------------------------------------------------------


from main import _init_paths
import torch
from lib.config import cfg, update_config
from lib.dataset import VQARADFeatureDataset
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from lib.utils.create_dictionary import Dictionary
import argparse
import torch.nn.functional as F
from datetime import datetime
from lib.utils import utils
from lib.language.classify_question import classify_model

# set  TOKENIZERS_PARALLELISM=(true | false)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
BERT_MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 8

def parse_args():
    parser = argparse.ArgumentParser(description="Type classifier")
    # GPU config
    parser.add_argument(
            "--cfg",
            help="decide which cfg to use",
            required=False,
            default="./configs/qcr_pubmedclipRN50_ae_rad_16batchsize_withtfidf_nondeterministic.yaml",
            type=str,
            )
    # GPU config
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')
    parser.add_argument('--seed', type=int, default=5
                        , help='random seed for gpu. Default:5')
    args = parser.parse_args()
    return args


# Evaluation
def evaluate(model, dataloader, logger, device):
    score = 0
    number =0
    model.eval()
    with torch.no_grad():
        for i,row in enumerate(dataloader):
            # see lib/dataset/dataset_RAD_bert.py for eval data format
            image_data, question, target, answer_type, question_type, phrase_type, answer_target, _, _, _ = row
            answer_target = answer_target.to(device)
            output = net(question)
            pred = output.data.max(1)[1]
            correct = (pred == answer_target).data.cpu().sum()

            output = model(question)
            pred = output.data.max(1)[1]
            correct = pred.eq(answer_target.data).cpu().sum()
            score += correct.item()
            number += len(answer_target)

        score = score / number * 100.

    logger.info('[Validate] Val_Acc:{:.6f}%'.format(score))
    return score


if __name__=='__main__':
    args = parse_args()
    update_config(cfg, args)
    dataroot = cfg.DATASET.DATA_DIR
    # # set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    # set to cpu for clearer debugging message
    # device = torch.device("cpu")

    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # d = Dictionary.load_from_file(os.path.join(dataroot, 'dictionary.pkl'))
    d = Dictionary.load_from_model_name(BERT_MODEL_NAME)
    train_dataset = VQARADFeatureDataset('train', cfg, d, dataroot=dataroot)
    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

    val_dataset = VQARADFeatureDataset('test', cfg, d, dataroot=dataroot)
    val_data = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    # net = classify_model(d.ntoken, os.path.join(dataroot, 'glove6b_init_300d.npy'))
    # TODO: 改成用 BERT embedding 訓練 (DONE 2023/06/02)
    net = classify_model(bert_model_name=BERT_MODEL_NAME,
                         in_dim=768)
    # fix
    net = net.to(device)

    run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
    ckpt_path = os.path.join('./log', run_timestamp)
    utils.create_dir(ckpt_path)
    model_path = os.path.join(ckpt_path, "type_classifier_rad_bert.pth")
    # create logger
    logger = utils.Logger(os.path.join(ckpt_path, 'medVQA.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(net)
    logger.info(">>>The args is:")
    logger.info(args.__repr__())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    epochs = 200
    best_eval_score = 0
    best_epoch = 0
    for epoch in range(epochs):
        net.train()
        acc = 0.
        number_dataset = 0
        total_loss = 0
        for i, row in enumerate(train_data):
            image_data, question, target, answer_type, question_type, phrase_type, answer_target = row
            # question[0], answer_target = question[0].to(device), answer_target.to(device)
            # question" List[str]
            answer_target = answer_target.to(device)
            optimizer.zero_grad()
            output = net(question)
            loss = criterion(output, answer_target)
            loss.backward()
            optimizer.step()
            pred = output.data.max(1)[1]
            correct = (pred == answer_target).data.cpu().sum()

            acc += correct.item()
            number_dataset += len(answer_target)
            total_loss+= loss

        total_loss /= len(train_data)
        acc = acc/ number_dataset * 100.

        logger.info('-------[Epoch]:{}-------'.format(epoch))
        logger.info('[Train] Loss:{:.6f} , Train_Acc:{:.6f}%'.format(total_loss, acc
                                                                     ))
        # Evaluation
        if val_data is not None:
            eval_score = evaluate(net, val_data, logger, device)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_epoch = epoch
                utils.save_model(model_path, net, best_epoch, eval_score)
            logger.info('[Result] The best acc is {:.6f}% at epoch {}'.format(best_eval_score, best_epoch))
