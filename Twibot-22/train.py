from Model import MY_MODEL
from Dataset import Twibot20,MyDataset,MyDataset
import torch
from torch import nn
from utils import accuracy,init_weights,num_correct,save_model_except
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import os
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from dataset_tools import custom_collate_fn
from transformers import get_cosine_schedule_with_warmup
import argparse

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
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
    model = torch.load("./trained_model_param/all/deepseek/MY_MODEL.pt")
    all_predictions = []
    all_targets = []
    all_scores = []

    model.eval()
    with torch.no_grad():   # 关闭梯度计算
        for batch_index, batch_prompts, batch_labels in tqdm(test_dataloader):
            output = model(
                des=des_tensor,
                tweet=tweets_tensor,
                num_prop=num_prop,
                cat_prop=category_prop,
                pre_x=semantic_tensor,
                edge_index=edge_index,
                edge_type=edge_type,
                prompt=batch_prompts,
                idx=batch_index
            )

            # 如果 output 是 logits，使用 softmax 得到概率
            probs = torch.softmax(output, dim=1)

            # 预测标签
            predictions = torch.argmax(probs, dim=1)

            # bot 类概率，假设 bot = 1
            bot_scores = probs[:, 1]

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_labels.cpu().numpy())
            all_scores.extend(bot_scores.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_scores = np.array(all_scores)

    acc = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    bot_precision = precision_score(
        all_targets,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    bot_recall = recall_score(
        all_targets,
        all_predictions,
        pos_label=1,
        zero_division=0
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0
    )

    roc_auc = roc_auc_score(
        all_targets,
        all_scores
    )

    pr_auc = average_precision_score(
        all_targets,
        all_scores,
        pos_label=1
    )

    cm_norm = confusion_matrix(
        all_targets,
        all_predictions,
        labels=[0, 1],
        normalize='true'
    )

    print("Test set results:")
    print("test_accuracy = {:.4f}".format(acc))
    print("f1        = {:.4f}".format(f1))
    print("bot_precision = {:.4f}".format(bot_precision))
    print("bot_recall    = {:.4f}".format(bot_recall))
    print("macro_f1      = {:.4f}".format(macro_f1))
    print("roc_auc       = {:.4f}".format(roc_auc))
    print("pr_auc        = {:.4f}".format(pr_auc))
    print("confusion_matrix:")
    print(cm_norm)

    return acc, f1


def main(seed):
    best_acc = 0
    best_f1 = 0
    best_epoch = 0
    model=MY_MODEL(
                llm_path=llm_path,
                llm_size=llm_size,
                llm_freeze_param=True,
                in_num_prop_size=5,
                in_cat_prop_size=3,
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
         num_training_steps=num_warmup_steps
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
    print('best val accuracy = ',best_acc,flush=True)
    print('best val f1 score = ',best_f1,flush=True)
    print('best val epoch = ',best_epoch,flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    args = parser.parse_args()
    set_seed(args.seed)

    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    device = 'cuda'
    embedding_size,dropout,lr,weight_decay=32,0.1,4e-4,5e-2  #4e-4
    epochs=25
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

    dataset=Twibot20(root=root,device=device)
    des_tensor,tweets_tensor,semantic_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.get_data(semantic_type='gpt')

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
