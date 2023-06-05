# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         train
# Description:
# Author:       Boliu.Kelvin, Sedigheh Eslami
#-------------------------------------------------------------------------------
import os
import time
import torch
from datetime import datetime
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.utils import utils

OPTIM_MAP = {
    'ADAMAX': torch.optim.Adamax,
    'ADAM': torch.optim.Adam,
    'SGD': torch.optim.SGD,
    'ADAMW': torch.optim.AdamW
}


def compute_score_with_logits(logits, labels):
    if labels.shape[0] == 0:  # sometimes, all samples in the batch are either open or close
        # hence, the labels and logits is empty
        scores = torch.zeros(*labels.size()).to(logits.device)
        return scores
    func = torch.nn.Softmax(dim=1)
    logits = func(logits)
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp


# Train phase
def train(cfg,
          model,
          question_model,
          train_loader,
          eval_loader,
          n_unique_close,
          device,
          s_opt=None,
          s_epoch=0):
    # model 是 BAN_Model
    # question_model 是 question_classify
    tblog_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboardlogs")
    if not os.path.exists(tblog_dir):
        os.makedirs(tblog_dir)
    writer = SummaryWriter(log_dir=tblog_dir)

    model = model.to(device)
    question_model = question_model.to(device)
    # create packet for output
    utils.create_dir(cfg.OUTPUT_DIR)
    # for every train, create a packet for saving .pth and .log
    ckpt_path = os.path.join(cfg.OUTPUT_DIR, cfg.NAME)
    utils.create_dir(ckpt_path)
    # create logger
    logger = utils.Logger(os.path.join(ckpt_path, 'medVQA.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(model)

    # transformers 需要 warmup, lr 也不可以太大
    """
    credit:
    Eric: 2023/06/03 added warmup for BERT
    Nana: 2023/06/03 revised to have 2 optimizers for BERT and BAN
    """
    ban_model = model
    bert_model = model.q_emb_model.model
    # =========== BAN model optimizer ===============
    no_bert_params = []
    for name, param in ban_model.named_parameters():
        if not name.startswith("q_emb_model.model"):
            no_bert_params.append(param)

    ban_optimizer = OPTIM_MAP[cfg.TRAIN.BAN_OPTIMIZER.TYPE](params=no_bert_params,
                                                            lr=cfg.TRAIN.BAN_OPTIMIZER.BASE_LR)

    # =========== BERT model optimizer ===============

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    weight_decay = 1e-2
    param_optimizer = list(bert_model.named_parameters())
    optimizer_grouped_parameters = [{
        'params':
        [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
        'weight_decay':
        weight_decay
    }, {
        'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
        'weight_decay':
        0.0
    }]
    bert_optimizer = OPTIM_MAP[cfg.TRAIN.BERT_OPTIMIZER.TYPE](params=optimizer_grouped_parameters,
                                                              lr=cfg.TRAIN.BERT_OPTIMIZER.BASE_LR)
    global_step = int(cfg.TRAIN.N_EPOCH * len(train_loader))
    num_warmup_steps = int(global_step * 0.1)
    logger.info(f"global_step: {global_step}, num_warmup_steps: {num_warmup_steps}")
    bert_lr_scheduler = get_linear_schedule_with_warmup(bert_optimizer,
                                                        num_warmup_steps=int(global_step * 0.1),
                                                        num_training_steps=global_step)

    logger.info(f"ban_optimizer type: {cfg.TRAIN.BAN_OPTIMIZER.TYPE}")
    if not cfg.DATASET.FIX_EMB_MODEL:
        logger.info(f"bert_optimizer type: {cfg.TRAIN.BERT_OPTIMIZER.TYPE}")
    # =======================================================
    # Loss function
    if cfg.LOSS.LOSS_TYPE == "BCELogits":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif cfg.LOSS.LOSS_TYPE == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{cfg.LOSS.LOSS_TYPE} loss not supported!")

    ae_criterion = torch.nn.MSELoss()

    best_eval_score = 0
    best_epoch = 0
    # Epoch passing in training phase
    for epoch in range(s_epoch, cfg.TRAIN.N_EPOCH):

        total_loss = 0
        total_open_loss = 0
        total_close_loss = 0
        train_score = 0
        number = 0
        open_cnt = 0
        close_cnt = 0
        model.train()

        # Predicting and computing score
        for i, (v, q, a, answer_type, question_type, phrase_type,
                answer_target) in enumerate(train_loader):
            ban_optimizer.zero_grad()
            bert_optimizer.zero_grad()
            if cfg.TRAIN.VISION.MAML:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if cfg.TRAIN.VISION.AUTOENCODER:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if cfg.TRAIN.VISION.CLIP:
                if cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
                    v[2] = v[2].reshape(v[2].shape[0], 3, 288, 288)
                else:
                    v[2] = v[2].reshape(v[2].shape[0], 3, 250, 250)
                v[2] = v[2].to(device)
            if cfg.TRAIN.VISION.OTHER_MODEL:
                v = v.to(device)

            # prepare bert input
            # q = List[str]
            a = a.to(device)
            answer_target = answer_target.to(device)
            if cfg.TRAIN.VISION.AUTOENCODER:
                last_output_close, last_output_open, a_close, a_open, decoder = model(
                    v, q, a, answer_target)
            else:
                last_output_close, last_output_open, a_close, a_open = model(v, q, a, answer_target)

            preds_close, preds_open = model.classify(last_output_close, last_output_open)

            #loss
            if cfg.LOSS.LOSS_TYPE == "BCELogits":
                loss_close = criterion(preds_close.float(), a_close)
                loss_open = criterion(preds_open.float(), a_open)
            elif cfg.LOSS.LOSS_TYPE == "CrossEntropy":
                loss_close = criterion(preds_close.float(), torch.max(a_close, 1)[1])
                loss_open = criterion(preds_open.float(), torch.max(a_open, 1)[1])
            if torch.isnan(loss_open):
                assert a_open.shape[0] == 0
                loss_open = torch.tensor([0.0]).to(device)
            if torch.isnan(loss_close):
                assert a_close.shape[0] == 0
                loss_close = torch.tensor([0.0]).to(device)
            loss = loss_close + loss_open
            if cfg.TRAIN.VISION.AUTOENCODER:
                loss_ae = ae_criterion(v[1], decoder)
                loss = loss + (loss_ae * cfg.TRAIN.VISION.AE_ALPHA)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            bert_optimizer.step()
            bert_lr_scheduler.step()
            ban_optimizer.step()
            #compute the acc for open and close
            batch_close_score = compute_score_with_logits(preds_close, a_close.data).sum()
            batch_open_score = compute_score_with_logits(preds_open, a_open.data).sum()
            total_open_loss += loss_open.item()
            total_close_loss += loss_close.item()
            total_loss += loss.item()
            train_score += batch_close_score + batch_open_score
            # number+= q[0].shape[0]
            number += len(q)  # batch size
            open_cnt += preds_open.shape[0]
            close_cnt += preds_close.shape[0]

        total_loss /= len(train_loader)
        total_open_loss /= open_cnt
        total_close_loss /= close_cnt
        train_score = 100 * train_score / number
        logger.info('-------[Epoch]:{}-------'.format(epoch))
        logger.info('[Train] Loss:{:.6f} , Train_Acc:{:.6f}%'.format(total_loss, train_score))
        logger.info('[Train] Loss_Open:{:.6f} , Loss_Close:{:.6f}%'.format(
            total_open_loss, total_close_loss))
        logger.info('[Current lr (BAN)]:{:.10f}'.format(ban_optimizer.param_groups[0]['lr']))
        logger.info('[Current lr (BERT)]:{:.10f}'.format(bert_optimizer.param_groups[0]['lr']))

        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Loss_Open/train", total_open_loss, epoch)
        writer.add_scalar("Loss_Close/train", total_close_loss, epoch)
        writer.add_scalar("Accuracy/train", train_score, epoch)

        # Evaluation
        if eval_loader is not None:
            eval_score, open_score, close_score = evaluate_classifier(model, question_model,
                                                                      eval_loader, cfg,
                                                                      n_unique_close, device,
                                                                      logger)
            if eval_score > best_eval_score:
                # clean all models
                utils.keep_last_n_models(ckpt_path, n=0)
                # save current best
                best_eval_score = eval_score
                best_epoch = epoch
                # Save the best acc epoch
                model_path = os.path.join(ckpt_path, f'{best_epoch}_best.pth')
                utils.save_model(model_path,
                                 model,
                                 best_epoch,
                                 eval_score,
                                 open_score,
                                 close_score,
                                 ban_optimizer=ban_optimizer,
                                 bert_optimizer=bert_optimizer)


            logger.info('[Result] The best acc is {:.6f}% at epoch {}'.format(
                best_eval_score, best_epoch))
            writer.add_scalar("Accuracy/val", eval_score, epoch)
            writer.add_scalar("Accuracy/val/open", open_score, epoch)
            writer.add_scalar("Accuracy/val/close", close_score, epoch)


# Evaluation
def evaluate_classifier(model, pretrained_model, dataloader, cfg, n_unique_close, device, logger):
    score = 0
    total = 0
    open_ended = 0.  #'OPEN'
    score_open = 0.

    closed_ended = 0.  #'CLOSED'
    score_close = 0.
    model.eval()

    with torch.no_grad():
        for i, (v, q, a, answer_type, question_type, phrase_type, answer_target, _, _,
                _) in enumerate(dataloader):
            if cfg.TRAIN.VISION.MAML:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if cfg.TRAIN.VISION.AUTOENCODER:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if cfg.TRAIN.VISION.CLIP:
                if cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
                    v[2] = v[2].reshape(v[2].shape[0], 3, 288, 288)
                else:
                    v[2] = v[2].reshape(v[2].shape[0], 3, 250, 250)
                v[2] = v[2].to(device)
            if cfg.TRAIN.VISION.OTHER_MODEL:
                v = v.to(device)

            #q[0] = q[0].to(device)
            # if cfg.TRAIN.QUESTION.CLIP:
            #     q[1] = q[1].to(device)
            a = a.to(device)

            if cfg.TRAIN.VISION.AUTOENCODER:
                last_output_close, last_output_open, a_close, a_open, decoder, _, _ = model.forward_classify(
                    v, q, a, pretrained_model, n_unique_close)
            else:
                last_output_close, last_output_open, a_close, a_open, _, _ = model.forward_classify(
                    v, q, a, pretrained_model, n_unique_close)

            preds_close, preds_open = model.classify(last_output_close, last_output_open)

            batch_close_score = 0.
            batch_open_score = 0.
            if preds_close.shape[0] != 0:
                batch_close_score = compute_score_with_logits(preds_close, a_close.data).sum()
            if preds_open.shape[0] != 0:
                batch_open_score = compute_score_with_logits(preds_open, a_open.data).sum()

            score += batch_close_score + batch_open_score

            size = len(q)  # batch size
            total += size  # batch number

            open_ended += preds_open.shape[0]
            score_open += batch_open_score

            closed_ended += preds_close.shape[0]
            score_close += batch_close_score

    try:
        score = 100 * score / total
    except ZeroDivisionError:
        score = 0
    try:
        open_score = 100 * score_open / open_ended
    except ZeroDivisionError:
        open_score = 0
    try:
        close_score = 100 * score_close / closed_ended
    except ZeroDivisionError:
        close_score = 0
    print(total, open_ended, closed_ended)
    logger.info('[Validate] Val_Acc:{:.6f}%  |  Open_ACC:{:.6f}%   |  Close_ACC:{:.6f}%'.format(
        score, open_score, close_score))
    return score, open_score, close_score
