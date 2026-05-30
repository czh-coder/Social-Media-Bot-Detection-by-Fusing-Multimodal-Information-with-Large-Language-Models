from Model import MY_MODEL
from Dataset import Twibot20,MyDataset
import torch
from torch import nn
from utils import accuracy,init_weights,num_correct,save_model_except
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc
import os
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from dataset_tools import custom_collate_fn
from transformers import get_cosine_schedule_with_warmup
import argparse

def set_seed(seed):
    # 设置 Python 的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(epoch,model,criterion,optimizer,scheduler):
    all_predictions = []
    all_targets = []
    epoch_loss = 0.0
    i=0
    model.train()
    for batch_index,batch_prompts,batch_labels in tqdm(train_dataloader):
        output = model(des=des_tensor,
                       tweet=tweets_tensor,
                       num_prop=num_prop,
                       cat_prop=category_prop,
                       pre_x=semantic_tensor,
                       edge_index=edge_index,
                       edge_type=edge_type,
                       prompt=batch_prompts,
                       idx=batch_index)

        loss_train = criterion(output, batch_labels)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
            
        # 更新学习率
        scheduler.step()

        _, predictions = torch.max(output, dim=1)  # 获取预测的类别
        # 收集预测结果和真实标签
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(batch_labels.cpu().numpy())
        
        epoch_loss += loss_train.item()
        i+=1

    acc = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    avg_loss = epoch_loss / len(train_dataloader)
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(avg_loss),
        'acc_train: {:.4f}'.format(acc),
        'f1_train: {:.4f}'.format(f1),flush=True)

def val(model,criterion):
    all_predictions = []
    all_targets = []
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_index,batch_prompts,batch_labels in tqdm(val_dataloader):
            output = model(des=des_tensor,
                        tweet=tweets_tensor,
                        num_prop=num_prop,
                        cat_prop=category_prop,
                        pre_x=semantic_tensor,
                        edge_index=edge_index,
                        edge_type=edge_type,
                        prompt=batch_prompts,
                        idx=batch_index)

            loss_val = criterion(output, batch_labels)
            _, predictions = torch.max(output, dim=1)  # 获取预测的类别
            # 收集预测结果和真实标签
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
            epoch_loss += loss_val.item()

    avg_loss = epoch_loss / len(val_dataloader)
    acc = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    print("Val set results:",
          "val_loss= {:.4f}".format(avg_loss),
            "val_accuracy= {:.4f}".format(acc),
            "val_f1_score= {:.4f}".format(f1),
            flush=True)
    return acc,f1

def test():
    model = torch.load("./trained_model_param/all/gpt/MY_MODEL.pt")
    all_predictions = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for batch_index,batch_prompts,batch_labels in test_dataloader:
            output = model(des=des_tensor,
                        tweet=tweets_tensor,
                        num_prop=num_prop,
                        cat_prop=category_prop,
                        pre_x=semantic_tensor,
                        edge_index=edge_index,
                        edge_type=edge_type,
                        prompt=batch_prompts,
                        idx=batch_index)

            _, predictions = torch.max(output, dim=1)  # 获取预测的类别
            # 收集预测结果和真实标签
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())

    all_targets_np = np.array(all_targets)
    all_predictions_np = np.array(all_predictions)

    twibot_len = len(test_idx_twibot)
    cresci_len = len(test_idx_cresci)
    # 生成all_targets_np中Twibot20样本的索引（0到twibot_len-1）
    test_idx_twibot_new = np.arange(twibot_len)
    # 生成all_targets_np中Cresci15样本的索引（twibot_len到twibot_len+cresci_len-1）
    test_idx_cresci_new = np.arange(twibot_len, twibot_len + cresci_len)

    acc = accuracy_score(all_targets_np, all_predictions_np)
    f1 = f1_score(all_targets_np, all_predictions_np)

    acc_twibot = accuracy_score(all_targets_np[test_idx_twibot_new], all_predictions_np[test_idx_twibot_new])
    f1_twibot = f1_score(all_targets_np[test_idx_twibot_new], all_predictions_np[test_idx_twibot_new])
    acc_cresci = accuracy_score(all_targets_np[test_idx_cresci_new], all_predictions_np[test_idx_cresci_new])
    f1_cresci = f1_score(all_targets_np[test_idx_cresci_new], all_predictions_np[test_idx_cresci_new])

    print("Test set results:",
            "test_accuracy= {:.4f}".format(acc),
            "f1_score= {:.4f}".format(f1),
            "test_accuracy_twibot= {:.4f}".format(acc_twibot),
            "f1_score_twibot= {:.4f}".format(f1_twibot),
            "test_accuracy_cresci= {:.4f}".format(acc_cresci),
            "f1_score_cresci= {:.4f}".format(f1_cresci),
            flush=True)
    return acc,f1,acc_twibot,f1_twibot,acc_cresci,f1_cresci

def main(seed):
    best_acc = 0
    best_f1 = 0
    best_epoch = 0

    best_acc_twibot = 0
    best_f1_twibot = 0
    best_acc_cresci = 0
    best_f1_cresci = 0

    model=MY_MODEL(
                llm_path=llm_path,
                llm_size=llm_size,
                llm_freeze_param=True,
                in_num_prop_size=5,
                in_cat_prop_size=1,
                out_size=embedding_size,
                fusion_type='project_concat').to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                        lr=lr,weight_decay=weight_decay)
    # 定义学习率调度器
    num_training_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(0.15 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
         optimizer,
         num_warmup_steps=num_warmup_steps,  # 15% warmup
         num_training_steps=num_training_steps
    )
    for epoch in range(epochs):
        # print('epoch:',epoch+1,flush=True)

        train(epoch,model,criterion,optimizer,scheduler)
        

        acc,f1 = val(model,criterion)
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_epoch = epoch+1

            torch.save(model, "./trained_model_param/all/gpt/MY_MODEL.pt")
    print('seed = ',seed,flush=True)
    print('best test accuracy = ',best_acc,flush=True)
    print('best test f1 score = ',best_f1,flush=True)
    print('best test epoch = ',best_epoch,flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    args = parser.parse_args()
    set_seed(args.seed)

    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    device = 'cuda'
    embedding_size,dropout,lr,weight_decay=32,0.1,2e-4,5e-2
    epochs=50
    batch_size = 256
    # accumulation_steps = 64
    root = '../data/'
    llm_size = '1B'

    if llm_size == '1B':
        llm_path = '/home/czh/models/Llama-3.2-1B'
    elif llm_size == '3B':
        llm_path = '/home/czh/models/Llama-3.2-3B'
    elif llm_size == '8B':
        llm_path = '/home/czh/models/meta-llama/Llama-3.1-8B'

    test_idx_twibot = torch.load("/mnt/data_1/czh/projects/data/Cresci15-Twibot20/Twibot20/test_idx.pt")
    test_idx_cresci = torch.load("/mnt/data_1/czh/projects/data/Cresci15-Twibot20/Cresci15/test_idx.pt")

    dataset=Twibot20(root=root,device=device)
    des_tensor,tweets_tensor,semantic_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.get_data()

    train_dataset = MyDataset(train_idx,labels[train_idx])
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last = False,
        # collate_fn=custom_collate_fn
    )
    val_dataset = MyDataset(val_idx,labels[val_idx])
    val_dataloader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last = False,
        # collate_fn=custom_collate_fn
    )
    test_dataset = MyDataset(test_idx,labels[test_idx])
    test_dataloader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=batch_size,
        drop_last = False,
        # collate_fn=custom_collate_fn
    )

    main(args.seed)
    test()
