import torch
from torch import nn
from torch.optim import AdamW

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def num_correct(output,labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)

def save_model_except(module_name, model, path):
    """
    保存模型参数，但排除指定模块的参数
    
    参数:
    module_name (str): 要排除的模块名称（如"encoder"）
    model (torch.nn.Module): 模型
    path (str): 保存路径
    """
    # 获取完整状态字典
    state_dict = model.state_dict()
    
    # 创建新的状态字典，排除指定模块
    filtered_dict = {
        k: v for k, v in state_dict.items() 
        if not k.startswith(f"{module_name}.")
    }
    
    # 保存过滤后的字典
    torch.save(filtered_dict, path)
    print(f"模型已保存至 {path}，排除模块 {module_name}")

def get_optimizer(model,name='AdamW'):
    if name == 'AdamW':
        optimizer = AdamW([
            {"params": model.graph_encoder.parameters(), "lr": 1e-5},
            {"params": model.text_encoder.parameters(), "lr": 1e-5},
            {"params": model.metadata_encoder.parameters(), "lr": 1e-5},
            {"params": model.fusion.parameters(), "lr": 5e-5},
            {"params": model.mapping_m.parameters(), "lr": 1e-4},
            {"params": model.mapping_t.parameters(), "lr": 1e-4},
            {"params": model.mapping_g.parameters(), "lr": 1e-4},
            {"params": model.mapping_f.parameters(), "lr": 1e-4},
            {"params": model.align_m.parameters(), "lr": 1e-4},
            {"params": model.align_t.parameters(), "lr": 1e-4},
            {"params": model.align_g.parameters(), "lr": 1e-4},
            {"params": model.align_f.parameters(), "lr": 1e-4},
        ], weight_decay=0.01)
        
