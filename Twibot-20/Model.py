import torch
from torch import nn
from torch_geometric.nn import RGCNConv,FastRGCNConv,GCNConv,GATConv
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForSequenceClassification,AutoModel,AutoConfig
from torch_geometric.nn.models import MLP
from model.modeling_llama import LlamaForCausalLM,LlamaForSequenceClassification
from utils import init_weights

def mask_by_len(input, lens, fill_value=0):
    '''
    input: shape = [N, D]
    lens: shape = [N]
    '''
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input


class PreToken_Train:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instruction,max_length):
        self.tokenizer.padding_side = 'right'# 设定填充（padding）的方向为右边，这意味着较短的文本会在末尾添加填充
        self.tokenizer.truncation_side = 'right'# 设置截断方向为左边，文本会从左边开始截断以符合最大长度要求
        instruction_tokens = self.tokenizer(text=instruction,
                                            truncation=True,
                                            max_length=max_length,
                                            padding="max_length",
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            return_attention_mask=True)
        
        return instruction_tokens    

class BotRGCN(nn.Module):
    def __init__(self,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.3):
        super(BotRGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension/4)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        # self.linear_output2=nn.Linear(embedding_dimension,2)
        
        
        
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        x=torch.cat((d,t,n,c),dim=1)
        # x=torch.cat((n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        # x=self.linear_output2(x)
            
        return x

#-----------------------------------------------------------------------------------

class FeatureFusion(nn.Module):
    def __init__(self, dim1, dim2, hidden_size, output_size, fusion_type='concat', 
                 activation='relu', dropout=0.1):
        super().__init__()
        self.fusion_type = fusion_type
        
        # 选择激活函数
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'leakyRelu':
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.Identity()  # 无激活
            
        self.dropout = nn.Dropout(dropout)
        
        if fusion_type == 'project_concat':
            # 方法1：先映射到相同维度再拼接
            self.mlp1 = nn.Sequential(
                nn.Linear(dim1, hidden_size),
                self.act,
                nn.LayerNorm(hidden_size),
                self.dropout
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(dim2, hidden_size),
                self.act,
                nn.LayerNorm(hidden_size),
                self.dropout
            )
            self.out = nn.Sequential(
                nn.Linear(hidden_size*2, output_size),
                self.act,
                nn.LayerNorm(output_size),
                self.dropout
            )
            
        elif fusion_type == 'concat':
            # 方法2：直接拼接
            self.out = nn.Sequential(
                nn.Linear(dim1 + dim2, output_size),
                self.act,
                nn.LayerNorm(output_size),
                self.dropout
            )
            
        elif fusion_type == 'gated':
            # 门控融合
            self.mlp1 = nn.Sequential(
                nn.Linear(dim1, hidden_size),
                self.act,
                nn.LayerNorm(hidden_size),
                self.dropout
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(dim2, hidden_size),
                self.act,
                nn.LayerNorm(hidden_size),
                self.dropout
            )
            self.gate = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                self.act,
                nn.LayerNorm(hidden_size),
                self.dropout,
                nn.Sigmoid()
            )
            self.out = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                self.act,
                nn.LayerNorm(output_size),
                self.dropout
            )
            
        elif fusion_type == 'attention':
            # 注意力融合
            self.mlp1 = nn.Sequential(
                nn.Linear(dim1, hidden_size),
                self.act,
                nn.LayerNorm(hidden_size),
                self.dropout
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(dim2, hidden_size),
                self.act,
                nn.LayerNorm(hidden_size),
                self.dropout
            )
            self.attn = nn.Sequential(
                nn.Linear(hidden_size*2, hidden_size),
                self.act,
                nn.LayerNorm(hidden_size),
                self.dropout,
                nn.Linear(hidden_size, 1),
                nn.Softmax(dim=1)
            )
            self.out = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                self.act,
                nn.LayerNorm(output_size),
                self.dropout
            )
            
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
            
    def forward(self, x1, x2):
        if self.fusion_type == 'project_concat':
            h1 = self.mlp1(x1)
            h2 = self.mlp2(x2)
            return self.out(torch.cat([h1, h2], dim=1))
            
        elif self.fusion_type == 'concat':
            return self.out(torch.cat([x1, x2], dim=1))
            
        elif self.fusion_type == 'gated':
            h1 = self.mlp1(x1)
            h2 = self.mlp2(x2)
            g = self.gate(torch.cat([h1, h2], dim=1))
            fused = g * h1 + (1 - g) * h2
            return self.out(fused)
            
        elif self.fusion_type == 'attention':
            h1 = self.mlp1(x1)
            h2 = self.mlp2(x2)
            a = self.attn(torch.cat([h1, h2], dim=1))
            fused = a * h1 + (1 - a) * h2
            return self.out(fused)

class FeatureFusion2(nn.Module):
    def __init__(self, dim1, dim2, dim3,hidden_size, output_size, fusion_type='concat', 
                 activation='relu', dropout=0.1):
        super().__init__()
        self.fusion_type = fusion_type
        
        # 选择激活函数
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'leakyRelu':
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.Identity()  # 无激活
            
        self.dropout = nn.Dropout(dropout)
        
        if fusion_type == 'project_concat':
            # 方法1：先映射到相同维度再拼接
            self.mlp1 = nn.Sequential(
                nn.Linear(dim1, hidden_size),
                self.act,
                nn.LayerNorm(hidden_size),
                self.dropout
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(dim2, hidden_size),
                self.act,
                nn.LayerNorm(hidden_size),
                self.dropout
            )
            self.mlp3 = nn.Sequential(
                nn.Linear(dim3, hidden_size),
                self.act,
                nn.LayerNorm(hidden_size),
                self.dropout
            )
            self.out = nn.Sequential(
                nn.Linear(hidden_size*3, output_size),
                self.act,
                nn.LayerNorm(output_size),
                self.dropout
            )
            
        elif fusion_type == 'concat':
            # 方法2：直接拼接
            self.out = nn.Sequential(
                nn.Linear(dim1 + dim2 + dim3, output_size),
                self.act,
                nn.LayerNorm(output_size),
                self.dropout
            )  
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
            
    def forward(self, x1, x2, x3):
        if self.fusion_type == 'project_concat':
            h1 = self.mlp1(x1)
            h2 = self.mlp2(x2)
            h3 = self.mlp3(x3)
            return self.out(torch.cat([h1, h2, h3], dim=1))
            
        elif self.fusion_type == 'concat':
            return self.out(torch.cat([x1, x2, x3], dim=1))

class FUSION_multimodality(nn.Module):
    def __init__(self,each_embed_size,num_heads=8,dropout=0.3):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=each_embed_size*3,
                                            num_heads=num_heads,
                                            dropout=dropout,
                                            batch_first=True)
        self.norm = nn.LayerNorm(each_embed_size*3)

    def forward(self,metadata_tensor,text_tensor,graph_tensor):
        x = torch.cat((metadata_tensor,text_tensor,graph_tensor),dim=1)
        x,attention = self.multihead_attention(x,x,x)
        x = self.norm(x)
        return x
    
#-------------------------------------------------------------------------------
class ModalityAttentionFusion(nn.Module):
    def __init__(self, num_modalities,embed_dim, num_heads=4, ff_dim=128, dropout=0.1):
        super(ModalityAttentionFusion, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities  # text, profile, topology

        # 模态类型编码（类似 BERT 的 segment embedding）
        # 给每个模态（text, profile, topology）分配一个可学习的“身份向量”
        self.modality_type_embedding = nn.Embedding(self.num_modalities, embed_dim)

        # Transformer 编码层（1 层或多层均可）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True  # [B, seq_len, D]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 输出投影（可选）
        # self.output_proj = nn.Linear(embed_dim, embed_dim)
        # 动态模态权重
        self.modality_weights = nn.Parameter(torch.ones(self.num_modalities))

    def forward(self, embeds):
        # meta_embed, text_embed, graph_embed: [B, D]
        B, D = embeds[0].size()

        # 堆叠模态嵌入：[B, 3, D]
        x = torch.stack(embeds, dim=1)

        # 添加模态类型编码
        # 生成一个 tensor [0, 1, 2] 表示每个模态的 ID
        modality_ids = torch.arange(self.num_modalities, device=x.device)  # [0,1,2]
        modality_type_emb = self.modality_type_embedding(modality_ids)  # [3, D]
        x = x + modality_type_emb.unsqueeze(0)  # [B, 3, D] + [1, 3, D]

        # Transformer 编码
        x = self.transformer_encoder(x)  # [B, 3, D]

        # 融合策略：mean pooling，也可以使用 [CLS] token
        # fused = x.mean(dim=1)  # [B, D]
        # 自适应调整模态重要性
        weights = F.softmax(self.modality_weights, dim=0)   # [3]
        fused = torch.sum(x * weights.view(1, -1, 1), dim=1)  # 加权求和

        # 可选线性映射
        # fused = self.output_proj(fused)

        return fused  # 融合后的表示 [B, D]

class ModalityAttentionFusion_without_type_embed(nn.Module):
    def __init__(self, num_modalities,embed_dim, num_heads=4, ff_dim=128, dropout=0.1):
        super(ModalityAttentionFusion_without_type_embed, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities  # text, profile, topology

        # 模态类型编码（类似 BERT 的 segment embedding）
        # 给每个模态（text, profile, topology）分配一个可学习的“身份向量”
        # self.modality_type_embedding = nn.Embedding(self.num_modalities, embed_dim)

        # Transformer 编码层（1 层或多层均可）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True  # [B, seq_len, D]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 输出投影（可选）
        # self.output_proj = nn.Linear(embed_dim, embed_dim)
        # 动态模态权重
        self.modality_weights = nn.Parameter(torch.ones(self.num_modalities))

    def forward(self, embeds):
        # meta_embed, text_embed, graph_embed: [B, D]
        B, D = embeds[0].size()

        # 堆叠模态嵌入：[B, 3, D]
        x = torch.stack(embeds, dim=1)

        # 添加模态类型编码
        # 生成一个 tensor [0, 1, 2] 表示每个模态的 ID
        # modality_ids = torch.arange(self.num_modalities, device=x.device)  # [0,1,2]
        # modality_type_emb = self.modality_type_embedding(modality_ids)  # [3, D]
        # x = x + modality_type_emb.unsqueeze(0)  # [B, 3, D] + [1, 3, D]

        # Transformer 编码
        x = self.transformer_encoder(x)  # [B, 3, D]

        # 融合策略：mean pooling，也可以使用 [CLS] token
        # fused = x.mean(dim=1)  # [B, D]
        # 自适应调整模态重要性
        weights = F.softmax(self.modality_weights, dim=0)   # [3]
        fused = torch.sum(x * weights.view(1, -1, 1), dim=1)  # 加权求和

        # 可选线性映射
        # fused = self.output_proj(fused)

        return fused  # 融合后的表示 [B, D]

class ModalityAttentionFusion_mean_pool(nn.Module):
    def __init__(self, num_modalities,embed_dim, num_heads=4, ff_dim=128, dropout=0.1):
        super(ModalityAttentionFusion_mean_pool, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities  # text, profile, topology

        # 模态类型编码（类似 BERT 的 segment embedding）
        # 给每个模态（text, profile, topology）分配一个可学习的“身份向量”
        self.modality_type_embedding = nn.Embedding(self.num_modalities, embed_dim)

        # Transformer 编码层（1 层或多层均可）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True  # [B, seq_len, D]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 输出投影（可选）
        # self.output_proj = nn.Linear(embed_dim, embed_dim)
        # 动态模态权重
        # self.modality_weights = nn.Parameter(torch.ones(self.num_modalities))

    def forward(self, embeds):
        # meta_embed, text_embed, graph_embed: [B, D]
        B, D = embeds[0].size()

        # 堆叠模态嵌入：[B, 3, D]
        x = torch.stack(embeds, dim=1)

        # 添加模态类型编码
        # 生成一个 tensor [0, 1, 2] 表示每个模态的 ID
        modality_ids = torch.arange(self.num_modalities, device=x.device)  # [0,1,2]
        modality_type_emb = self.modality_type_embedding(modality_ids)  # [3, D]
        x = x + modality_type_emb.unsqueeze(0)  # [B, 3, D] + [1, 3, D]

        # Transformer 编码
        x = self.transformer_encoder(x)  # [B, 3, D]

        # 融合策略：mean pooling，也可以使用 [CLS] token
        fused = x.mean(dim=1)  # [B, D]
        # 自适应调整模态重要性
        # weights = F.softmax(self.modality_weights, dim=0)   # [3]
        # fused = torch.sum(x * weights.view(1, -1, 1), dim=1)  # 加权求和

        # 可选线性映射
        # fused = self.output_proj(fused)

        return fused  # 融合后的表示 [B, D]
    
class ModalityAttentionFusion_max_pool(nn.Module):
    def __init__(self, num_modalities,embed_dim, num_heads=4, ff_dim=128, dropout=0.1):
        super(ModalityAttentionFusion_max_pool, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities  # text, profile, topology

        # 模态类型编码（类似 BERT 的 segment embedding）
        # 给每个模态（text, profile, topology）分配一个可学习的“身份向量”
        self.modality_type_embedding = nn.Embedding(self.num_modalities, embed_dim)

        # Transformer 编码层（1 层或多层均可）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True  # [B, seq_len, D]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 输出投影（可选）
        # self.output_proj = nn.Linear(embed_dim, embed_dim)
        # 动态模态权重
        # self.modality_weights = nn.Parameter(torch.ones(self.num_modalities))

    def forward(self, embeds):
        # meta_embed, text_embed, graph_embed: [B, D]
        B, D = embeds[0].size()

        # 堆叠模态嵌入：[B, 3, D]
        x = torch.stack(embeds, dim=1)

        # 添加模态类型编码
        # 生成一个 tensor [0, 1, 2] 表示每个模态的 ID
        modality_ids = torch.arange(self.num_modalities, device=x.device)  # [0,1,2]
        modality_type_emb = self.modality_type_embedding(modality_ids)  # [3, D]
        x = x + modality_type_emb.unsqueeze(0)  # [B, 3, D] + [1, 3, D]

        # Transformer 编码
        x = self.transformer_encoder(x)  # [B, 3, D]

        # 融合策略：mean pooling，也可以使用 [CLS] token
        fused,_ = torch.max(x, dim=1)  # [B, D]
        # 自适应调整模态重要性
        # weights = F.softmax(self.modality_weights, dim=0)   # [3]
        # fused = torch.sum(x * weights.view(1, -1, 1), dim=1)  # 加权求和

        # 可选线性映射
        # fused = self.output_proj(fused)

        return fused  # 融合后的表示 [B, D]
    
class ModalityAttentionFusion_min_pool(nn.Module):
    def __init__(self, num_modalities,embed_dim, num_heads=4, ff_dim=128, dropout=0.1):
        super(ModalityAttentionFusion_min_pool, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities  # text, profile, topology

        # 模态类型编码（类似 BERT 的 segment embedding）
        # 给每个模态（text, profile, topology）分配一个可学习的“身份向量”
        self.modality_type_embedding = nn.Embedding(self.num_modalities, embed_dim)

        # Transformer 编码层（1 层或多层均可）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True  # [B, seq_len, D]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 输出投影（可选）
        # self.output_proj = nn.Linear(embed_dim, embed_dim)
        # 动态模态权重
        # self.modality_weights = nn.Parameter(torch.ones(self.num_modalities))

    def forward(self, embeds):
        # meta_embed, text_embed, graph_embed: [B, D]
        B, D = embeds[0].size()

        # 堆叠模态嵌入：[B, 3, D]
        x = torch.stack(embeds, dim=1)

        # 添加模态类型编码
        # 生成一个 tensor [0, 1, 2] 表示每个模态的 ID
        modality_ids = torch.arange(self.num_modalities, device=x.device)  # [0,1,2]
        modality_type_emb = self.modality_type_embedding(modality_ids)  # [3, D]
        x = x + modality_type_emb.unsqueeze(0)  # [B, 3, D] + [1, 3, D]

        # Transformer 编码
        x = self.transformer_encoder(x)  # [B, 3, D]

        # 融合策略：mean pooling，也可以使用 [CLS] token
        fused,_ = torch.min(x, dim=1)  # [B, D]
        # 自适应调整模态重要性
        # weights = F.softmax(self.modality_weights, dim=0)   # [3]
        # fused = torch.sum(x * weights.view(1, -1, 1), dim=1)  # 加权求和

        # 可选线性映射
        # fused = self.output_proj(fused)

        return fused  # 融合后的表示 [B, D]

#--------------------------------------------------------------------------------

class RGCN(nn.Module):
    def __init__(self, dropout=0.1,out_embed_dim=128,num_prop_size = 5,num_category_size = 3):
        super().__init__()
        self.hidden_dim = 32*5
        self.num_prop_size = num_prop_size
        self.num_category_size = num_category_size

        self.linear_relu_num_prop = nn.Linear(self.num_prop_size, int(self.hidden_dim/5))
        self.linear_relu_num_categpry = nn.Linear(self.num_category_size, int(self.hidden_dim/5))
        self.linear_relu_des = nn.Linear(768, int(self.hidden_dim/5))
        self.linear_relu_text = nn.Linear(768, int(self.hidden_dim/5))
        self.linear_relu_tweet = nn.Linear(768, int(self.hidden_dim/5))
        self.linear_relu_input = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.conv1 = RGCNConv(self.hidden_dim, self.hidden_dim, num_relations=2)
        self.conv2 = RGCNConv(self.hidden_dim, self.hidden_dim, num_relations=2)
        self.linear_relu_output1 = nn.Linear(self.hidden_dim, out_embed_dim)
        # self.linear_output2 = nn.Linear(80, 2)

        self.ReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        n = self.dropout(self.ReLU(self.linear_relu_num_prop(num_prop)))
        c = self.dropout(self.ReLU(self.linear_relu_num_categpry(cat_prop)))
        d = self.dropout(self.ReLU(self.linear_relu_des(des)))
        tw = self.dropout(self.ReLU(self.linear_relu_text(tweet)))
        pre_t = self.dropout(self.ReLU(self.linear_relu_tweet(pre_x)))
        x = torch.cat((n,c,d,tw,pre_t), dim=1)
        x = self.linear_relu_input(x)
        x = self.dropout(self.ReLU(x))
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        x = self.dropout(self.ReLU(self.linear_relu_output1(x)))

        return x

class BotRGCN_with_semantic(nn.Module):
    def __init__(self,semantic_size=768,des_size=768,tweet_size=768,num_prop_size=5,cat_prop_size=3,embedding_dimension=128,dropout=0.3):
        super(BotRGCN_with_semantic, self).__init__()
        self.dropout = dropout
        self.linear_relu_des=nn.Sequential(
            nn.Linear(des_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop=nn.Sequential(
            nn.Linear(num_prop_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop=nn.Sequential(
            nn.Linear(cat_prop_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        self.linear_relu_semantic=nn.Sequential(
            nn.Linear(semantic_size,int(embedding_dimension)),
            nn.LeakyReLU()
        )
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension*5,embedding_dimension*5),
            nn.LeakyReLU()
        )
        
        self.rgcn=RGCNConv(embedding_dimension*5,embedding_dimension*5,num_relations=2)
        
        self.linear_relu_output1=nn.Sequential(
            nn.Linear(embedding_dimension*5,embedding_dimension),
            nn.LeakyReLU()
        )
        # self.linear_output2=nn.Linear(embedding_dimension,2)
        
        
        
    def forward(self,semantic,des,tweet,num_prop,cat_prop,edge_index,edge_type):
        d=self.linear_relu_des(des)
        t=self.linear_relu_tweet(tweet)
        n=self.linear_relu_num_prop(num_prop)
        c=self.linear_relu_cat_prop(cat_prop)
        pre_x=self.linear_relu_semantic(semantic)
        x=torch.cat((pre_x,d,t,n,c),dim=1)
        # x=torch.cat((n,c),dim=1)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output1(x)
        # x=self.linear_output2(x)
            
        return x

class MY_MODEL_no_llm(nn.Module):
    def __init__(self,
                 encoder_out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(embedding_dimension=encoder_out_size,dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=encoder_out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,
                                              hidden_size=metadata_hidden_size,
                                              output_size=encoder_out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)

        self.fusion = ModalityAttentionFusion(num_modalities=3,
                                              embed_dim=encoder_out_size,
                                            ff_dim=encoder_out_size*2,
                                           dropout=fusion_dropout)
        
        self.score = nn.Sequential(
                            nn.Linear(encoder_out_size*4, encoder_out_size*2),
                            nn.LeakyReLU(0.1),
                            nn.LayerNorm(encoder_out_size*2),
                            nn.Dropout(0.1),
                            nn.Linear(encoder_out_size*2, 2),
                        )

        if is_linear_init:
            self.linear_init()

    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        self.fusion.apply(init_weights)
        self.score.apply(init_weights)

    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        final_embed = torch.cat([meta_embed,text_embed,graph_embed,fusion_embed],dim=1)
        logits = self.score(final_embed)

        return logits

class MY_MODEL(nn.Module):
    def __init__(self,
                 llm_path,
                 llm_freeze_param,
                 out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 llm_size='1B',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=out_size,
                                     dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,hidden_size=metadata_hidden_size,output_size=out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)
        

        # self.llm_model,self.pretoken_model_train,self.word_embeddings,self.vocab_size = self.init_llm_model(llm_path,llm_freeze_param)

        self.llm_model,self.pretoken_model_train = self.init_llm_model(llm_path,llm_freeze_param)

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight # [32001,4096] # 32002
        # self.vocab_size = self.word_embeddings.shape[0] # 32002

        # self.mapping_m = nn.Linear(self.vocab_size + 1, 1)# 将词汇表从n+1维映射到1维 图2(c)步骤
        # self.mapping_t = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_g = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_f = nn.Linear(self.vocab_size + 1, 1)

        self.fusion = ModalityAttentionFusion(num_modalities=3,
                                            embed_dim=out_size,
                                            ff_dim=out_size*2,
                                           dropout=fusion_dropout)
        if llm_size == '1B':
            self.align_m = nn.Linear(out_size,2048)
            self.align_t = nn.Linear(out_size,2048)
            self.align_g = nn.Linear(out_size,2048)
            self.align_f = nn.Linear(out_size,2048)
        elif llm_size == '3B':
            self.align_m = nn.Linear(out_size,3072)
            self.align_t = nn.Linear(out_size,3072)
            self.align_g = nn.Linear(out_size,3072)
            self.align_f = nn.Linear(out_size,3072)
        elif llm_size == '8B':
            self.align_m = nn.Linear(out_size,4096)
            self.align_t = nn.Linear(out_size,4096)
            self.align_g = nn.Linear(out_size,4096)
            self.align_f = nn.Linear(out_size,4096)
        if is_linear_init:
            self.linear_init()
        
    def init_llm_model(self,llm_path,freeze_param):
        # LLM 模型初始化
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='right')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        new_tokens = ["<metadata>","<text>","<graph>","<fusion>"]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, torch_dtype=torch.bfloat16)

        llm_model.resize_token_embeddings(len(tokenizer))
        llm_model.config.pad_token_id = tokenizer.pad_token_id

        pretoken_model_train=PreToken_Train(tokenizer)

        if freeze_param:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
        for param in llm_model.score.parameters():
            param.requires_grad = True
        
        return llm_model,pretoken_model_train
    
    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        # init_weights(self.mapping_m)
        # init_weights(self.mapping_t)
        # init_weights(self.mapping_g)
        # init_weights(self.mapping_f)
        init_weights(self.align_m)
        init_weights(self.align_t)
        init_weights(self.align_g)
        init_weights(self.align_f)
    
    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,prompt,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        meta_embed_aligned = self.align_m(meta_embed)
        text_embed_aligned = self.align_t(text_embed)
        graph_embed_aligned = self.align_g(graph_embed)
        fusion_embed_aligned = self.align_f(fusion_embed)

        instruction_tokens = self.pretoken_model_train(prompt,64)

        # 确保所有张量在同一设备上
        instruction_tokens = instruction_tokens.to(self.device)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            
        with torch.amp.autocast("cuda",dtype=torch.float16):

            # graph_inputs_llm = fusion_embed # [batch_size, 4096]
            meta_inputs_llm = meta_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            text_inputs_llm = text_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            graph_inputs_llm = graph_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            fusion_inputs_llm = fusion_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]

            # llama_word_embeddings = self.word_embeddings.unsqueeze(0) # [1,32001,4096]
            # llama_word_embeddings = llama_word_embeddings.repeat(
            # graph_inputs_llm.size(0), 1, 1).to(self.device)# 沿第一个维度复制 llama_word_embeddings,维度变为[batch_size,32001,4096]

            # cat_embedding_m = self.get_cat_embedding(meta_inputs_llm,llama_word_embeddings,"m")
            # cat_embedding_t = self.get_cat_embedding(text_inputs_llm,llama_word_embeddings,"t")
            # cat_embedding_g = self.get_cat_embedding(graph_inputs_llm,llama_word_embeddings,"g")
            # cat_embedding_f = self.get_cat_embedding(fusion_inputs_llm,llama_word_embeddings,"f")

            inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,graph_inputs_llm,fusion_inputs_llm)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=instruction_tokens.attention_mask,
                    return_dict=True,
                )
            logits = outputs.logits

        return logits

    def get_cat_embedding(self,modality_embeddings,llama_word_embeddings,modality):
        cat_embedding = torch.cat([modality_embeddings, llama_word_embeddings], dim=1)# [batch_size,32002,4096]
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()# [batch_size,4096,32002] # .contiguous()：确保张量在内存中的存储是连续的，避免可能的数据存储问题
        if modality=='m':
            cat_embedding = self.mapping_m(cat_embedding)# [batch_size,4096,1] # self.mapping是将32002->1
        elif modality=='t':
            cat_embedding = self.mapping_t(cat_embedding)
        elif modality=='g':
            cat_embedding = self.mapping_g(cat_embedding)
        elif modality=='f':
            cat_embedding = self.mapping_f(cat_embedding)
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

        return cat_embedding

    def concat_instruction_modality_embeds(self,instruction_embeds,embedding_m,embedding_t,embedding_g,embedding_f):
        # <metadata>:33
        # <text>:38
        # <graph>:43
        # <fusion>:49
        front_embeds = instruction_embeds[:,:33,:]
        # embedding_m
        mid_embeds1 = instruction_embeds[:,34:38,:]
        # embedding_t
        mid_embeds2 = instruction_embeds[:,39:43,:]
        # embedding_g
        mid_embeds3 = instruction_embeds[:,44:49,:]
        # embedding_f
        back_embeds = instruction_embeds[:,50:,:]
        new_embeds = torch.cat([front_embeds,
                                embedding_m,
                                mid_embeds1,
                                embedding_t,
                                mid_embeds2,
                                embedding_g,
                                mid_embeds3,
                                embedding_f,
                                back_embeds], dim=1)

        return new_embeds
    
class MY_MODEL_without_TRM(nn.Module):
    def __init__(self,
                 llm_path,
                 llm_freeze_param,
                 out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 llm_size='1B',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=out_size,
                                     dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,hidden_size=metadata_hidden_size,output_size=out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)
        

        # self.llm_model,self.pretoken_model_train,self.word_embeddings,self.vocab_size = self.init_llm_model(llm_path,llm_freeze_param)

        self.llm_model,self.pretoken_model_train = self.init_llm_model(llm_path,llm_freeze_param)

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight # [32001,4096] # 32002
        # self.vocab_size = self.word_embeddings.shape[0] # 32002

        # self.mapping_m = nn.Linear(self.vocab_size + 1, 1)# 将词汇表从n+1维映射到1维 图2(c)步骤
        # self.mapping_t = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_g = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_f = nn.Linear(self.vocab_size + 1, 1)

        # self.fusion = ModalityAttentionFusion(num_modalities=3,
        #                                     embed_dim=out_size,
        #                                     ff_dim=out_size*2,
        #                                    dropout=fusion_dropout)
        if llm_size == '1B':
            self.align_m = nn.Linear(out_size,2048)
            self.align_t = nn.Linear(out_size,2048)
            self.align_g = nn.Linear(out_size,2048)
            self.align_f = nn.Linear(out_size,2048)
        elif llm_size == '3B':
            self.align_m = nn.Linear(out_size,3072)
            self.align_t = nn.Linear(out_size,3072)
            self.align_g = nn.Linear(out_size,3072)
            self.align_f = nn.Linear(out_size,3072)
        elif llm_size == '8B':
            self.align_m = nn.Linear(out_size,4096)
            self.align_t = nn.Linear(out_size,4096)
            self.align_g = nn.Linear(out_size,4096)
            self.align_f = nn.Linear(out_size,4096)
        if is_linear_init:
            self.linear_init()
        
    def init_llm_model(self,llm_path,freeze_param):
        # LLM 模型初始化
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='right')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        new_tokens = ["<metadata>","<text>","<graph>"]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, torch_dtype=torch.bfloat16)

        llm_model.resize_token_embeddings(len(tokenizer))
        llm_model.config.pad_token_id = tokenizer.pad_token_id

        pretoken_model_train=PreToken_Train(tokenizer)

        if freeze_param:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
        for param in llm_model.score.parameters():
            param.requires_grad = True
        
        return llm_model,pretoken_model_train
    
    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        # init_weights(self.mapping_m)
        # init_weights(self.mapping_t)
        # init_weights(self.mapping_g)
        # init_weights(self.mapping_f)
        init_weights(self.align_m)
        init_weights(self.align_t)
        init_weights(self.align_g)
        # init_weights(self.align_f)
    
    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,prompt,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        # fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        meta_embed_aligned = self.align_m(meta_embed)
        text_embed_aligned = self.align_t(text_embed)
        graph_embed_aligned = self.align_g(graph_embed)
        # fusion_embed_aligned = self.align_f(fusion_embed)

        instruction_tokens = self.pretoken_model_train(prompt,64)

        # 确保所有张量在同一设备上
        instruction_tokens = instruction_tokens.to(self.device)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            
        with torch.amp.autocast("cuda",dtype=torch.float16):

            # graph_inputs_llm = fusion_embed # [batch_size, 4096]
            meta_inputs_llm = meta_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            text_inputs_llm = text_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            graph_inputs_llm = graph_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            # fusion_inputs_llm = fusion_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]

            # llama_word_embeddings = self.word_embeddings.unsqueeze(0) # [1,32001,4096]
            # llama_word_embeddings = llama_word_embeddings.repeat(
            # graph_inputs_llm.size(0), 1, 1).to(self.device)# 沿第一个维度复制 llama_word_embeddings,维度变为[batch_size,32001,4096]

            # cat_embedding_m = self.get_cat_embedding(meta_inputs_llm,llama_word_embeddings,"m")
            # cat_embedding_t = self.get_cat_embedding(text_inputs_llm,llama_word_embeddings,"t")
            # cat_embedding_g = self.get_cat_embedding(graph_inputs_llm,llama_word_embeddings,"g")
            # cat_embedding_f = self.get_cat_embedding(fusion_inputs_llm,llama_word_embeddings,"f")

            inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,graph_inputs_llm)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=instruction_tokens.attention_mask,
                    return_dict=True,
                )
            logits = outputs.logits

        return logits

    def get_cat_embedding(self,modality_embeddings,llama_word_embeddings,modality):
        cat_embedding = torch.cat([modality_embeddings, llama_word_embeddings], dim=1)# [batch_size,32002,4096]
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()# [batch_size,4096,32002] # .contiguous()：确保张量在内存中的存储是连续的，避免可能的数据存储问题
        if modality=='m':
            cat_embedding = self.mapping_m(cat_embedding)# [batch_size,4096,1] # self.mapping是将32002->1
        elif modality=='t':
            cat_embedding = self.mapping_t(cat_embedding)
        elif modality=='g':
            cat_embedding = self.mapping_g(cat_embedding)
        elif modality=='f':
            cat_embedding = self.mapping_f(cat_embedding)
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

        return cat_embedding

    def concat_instruction_modality_embeds(self,instruction_embeds,embedding_m,embedding_t,embedding_g):
        # <metadata>:26
        # <text>:31
        # <graph>:36
        front_embeds = instruction_embeds[:,:26,:]
        # embedding_m
        mid_embeds1 = instruction_embeds[:,27:31,:]
        # embedding_t
        mid_embeds2 = instruction_embeds[:,32:36,:]
        # embedding_g
        mid_embeds3 = instruction_embeds[:,37:,:]

        new_embeds = torch.cat([front_embeds,
                                embedding_m,
                                mid_embeds1,
                                embedding_t,
                                mid_embeds2,
                                embedding_g,
                                mid_embeds3], dim=1)

        return new_embeds

class MY_MODEL_without_llm(nn.Module):
    def __init__(self,
                 encoder_out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=encoder_out_size,
                                     dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=encoder_out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,
                                              hidden_size=metadata_hidden_size,
                                              output_size=encoder_out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)

        self.fusion = ModalityAttentionFusion(num_modalities=3,
                                              embed_dim=encoder_out_size,
                                            ff_dim=encoder_out_size*2,
                                           dropout=fusion_dropout)

        self.score = nn.Sequential(
                            nn.Linear(encoder_out_size*4, encoder_out_size*2),
                            nn.LeakyReLU(0.1),
                            nn.LayerNorm(encoder_out_size*2),
                            nn.Dropout(0.1),
                            nn.Linear(encoder_out_size*2, 2),
                        )

        if is_linear_init:
            self.linear_init()

    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        self.fusion.apply(init_weights)
        self.score.apply(init_weights)

    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        final_embed = torch.cat([meta_embed,text_embed,graph_embed,fusion_embed],dim=1)
        logits = self.score(final_embed)

        return logits
    
class MY_MODEL_without_TRM_llm(nn.Module):
    def __init__(self,
                 encoder_out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=encoder_out_size,
                                     dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=encoder_out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,
                                              hidden_size=metadata_hidden_size,
                                              output_size=encoder_out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)

        # self.fusion = ModalityAttentionFusion(num_modalities=3,
        #                                       embed_dim=encoder_out_size,
        #                                     ff_dim=encoder_out_size*2,
        #                                    dropout=fusion_dropout)

        self.score = nn.Sequential(
                            nn.Linear(encoder_out_size*3, encoder_out_size*2),
                            nn.LeakyReLU(0.1),
                            nn.LayerNorm(encoder_out_size*2),
                            nn.Dropout(0.1),
                            nn.Linear(encoder_out_size*2, 2),
                        )

        if is_linear_init:
            self.linear_init()

    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        # self.fusion.apply(init_weights)
        self.score.apply(init_weights)

    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        # fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        final_embed = torch.cat([meta_embed,text_embed,graph_embed],dim=1)
        logits = self.score(final_embed)

        return logits

class MY_MODEL_without_topic_emotion(nn.Module):
    def __init__(self,
                 llm_path,
                 llm_freeze_param,
                 out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 llm_size='1B',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=out_size,
                                     dropout=gnn_dropout)
        self.text_encoder = FeatureFusion(dim1=768,
                                            dim2=768,
                                            hidden_size=text_hidden_size,
                                            output_size=out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,
                                              hidden_size=metadata_hidden_size,
                                              output_size=out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)
        

        # self.llm_model,self.pretoken_model_train,self.word_embeddings,self.vocab_size = self.init_llm_model(llm_path,llm_freeze_param)

        self.llm_model,self.pretoken_model_train = self.init_llm_model(llm_path,llm_freeze_param)

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight # [32001,4096] # 32002
        # self.vocab_size = self.word_embeddings.shape[0] # 32002

        # self.mapping_m = nn.Linear(self.vocab_size + 1, 1)# 将词汇表从n+1维映射到1维 图2(c)步骤
        # self.mapping_t = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_g = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_f = nn.Linear(self.vocab_size + 1, 1)

        self.fusion = ModalityAttentionFusion(num_modalities=3,
                                            embed_dim=out_size,
                                            ff_dim=out_size*2,
                                           dropout=fusion_dropout)
        if llm_size == '1B':
            self.align_m = nn.Linear(out_size,2048)
            self.align_t = nn.Linear(out_size,2048)
            self.align_g = nn.Linear(out_size,2048)
            self.align_f = nn.Linear(out_size,2048)
        elif llm_size == '3B':
            self.align_m = nn.Linear(out_size,3072)
            self.align_t = nn.Linear(out_size,3072)
            self.align_g = nn.Linear(out_size,3072)
            self.align_f = nn.Linear(out_size,3072)
        elif llm_size == '8B':
            self.align_m = nn.Linear(out_size,4096)
            self.align_t = nn.Linear(out_size,4096)
            self.align_g = nn.Linear(out_size,4096)
            self.align_f = nn.Linear(out_size,4096)
        if is_linear_init:
            self.linear_init()
        
    def init_llm_model(self,llm_path,freeze_param):
        # LLM 模型初始化
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='right')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        new_tokens = ["<metadata>","<text>","<graph>","<fusion>"]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, torch_dtype=torch.bfloat16)

        llm_model.resize_token_embeddings(len(tokenizer))
        llm_model.config.pad_token_id = tokenizer.pad_token_id

        pretoken_model_train=PreToken_Train(tokenizer)

        if freeze_param:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
        for param in llm_model.score.parameters():
            param.requires_grad = True
        
        return llm_model,pretoken_model_train
    
    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        # init_weights(self.mapping_m)
        # init_weights(self.mapping_t)
        # init_weights(self.mapping_g)
        # init_weights(self.mapping_f)
        init_weights(self.align_m)
        init_weights(self.align_t)
        init_weights(self.align_g)
        init_weights(self.align_f)
    
    def forward(self,des,tweet,num_prop,cat_prop,edge_index,edge_type,prompt,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        meta_embed_aligned = self.align_m(meta_embed)
        text_embed_aligned = self.align_t(text_embed)
        graph_embed_aligned = self.align_g(graph_embed)
        fusion_embed_aligned = self.align_f(fusion_embed)

        instruction_tokens = self.pretoken_model_train(prompt,64)

        # 确保所有张量在同一设备上
        instruction_tokens = instruction_tokens.to(self.device)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            
        with torch.amp.autocast("cuda",dtype=torch.float16):

            # graph_inputs_llm = fusion_embed # [batch_size, 4096]
            meta_inputs_llm = meta_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            text_inputs_llm = text_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            graph_inputs_llm = graph_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            fusion_inputs_llm = fusion_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]

            # llama_word_embeddings = self.word_embeddings.unsqueeze(0) # [1,32001,4096]
            # llama_word_embeddings = llama_word_embeddings.repeat(
            # graph_inputs_llm.size(0), 1, 1).to(self.device)# 沿第一个维度复制 llama_word_embeddings,维度变为[batch_size,32001,4096]

            # cat_embedding_m = self.get_cat_embedding(meta_inputs_llm,llama_word_embeddings,"m")
            # cat_embedding_t = self.get_cat_embedding(text_inputs_llm,llama_word_embeddings,"t")
            # cat_embedding_g = self.get_cat_embedding(graph_inputs_llm,llama_word_embeddings,"g")
            # cat_embedding_f = self.get_cat_embedding(fusion_inputs_llm,llama_word_embeddings,"f")

            inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,graph_inputs_llm,fusion_inputs_llm)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=instruction_tokens.attention_mask,
                    return_dict=True,
                )
            logits = outputs.logits

        return logits

    def get_cat_embedding(self,modality_embeddings,llama_word_embeddings,modality):
        cat_embedding = torch.cat([modality_embeddings, llama_word_embeddings], dim=1)# [batch_size,32002,4096]
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()# [batch_size,4096,32002] # .contiguous()：确保张量在内存中的存储是连续的，避免可能的数据存储问题
        if modality=='m':
            cat_embedding = self.mapping_m(cat_embedding)# [batch_size,4096,1] # self.mapping是将32002->1
        elif modality=='t':
            cat_embedding = self.mapping_t(cat_embedding)
        elif modality=='g':
            cat_embedding = self.mapping_g(cat_embedding)
        elif modality=='f':
            cat_embedding = self.mapping_f(cat_embedding)
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

        return cat_embedding

    def concat_instruction_modality_embeds(self,instruction_embeds,embedding_m,embedding_t,embedding_g,embedding_f):
        # <metadata>:33
        # <text>:38
        # <graph>:43
        # <fusion>:49
        front_embeds = instruction_embeds[:,:33,:]
        # embedding_m
        mid_embeds1 = instruction_embeds[:,34:38,:]
        # embedding_t
        mid_embeds2 = instruction_embeds[:,39:43,:]
        # embedding_g
        mid_embeds3 = instruction_embeds[:,44:49,:]
        # embedding_f
        back_embeds = instruction_embeds[:,50:,:]
        new_embeds = torch.cat([front_embeds,
                                embedding_m,
                                mid_embeds1,
                                embedding_t,
                                mid_embeds2,
                                embedding_g,
                                mid_embeds3,
                                embedding_f,
                                back_embeds], dim=1)

        return new_embeds

class MY_MODEL_TRM2(nn.Module):
    def __init__(self,
                 llm_path,
                 llm_freeze_param,
                 out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 llm_size='1B',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=out_size,
                                     dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,hidden_size=metadata_hidden_size,output_size=out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)
        

        # self.llm_model,self.pretoken_model_train,self.word_embeddings,self.vocab_size = self.init_llm_model(llm_path,llm_freeze_param)

        self.llm_model,self.pretoken_model_train = self.init_llm_model(llm_path,llm_freeze_param)

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight # [32001,4096] # 32002
        # self.vocab_size = self.word_embeddings.shape[0] # 32002

        # self.mapping_m = nn.Linear(self.vocab_size + 1, 1)# 将词汇表从n+1维映射到1维 图2(c)步骤
        # self.mapping_t = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_g = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_f = nn.Linear(self.vocab_size + 1, 1)

        self.fusion = ModalityAttentionFusion_without_type_embed(num_modalities=3,
                                            embed_dim=out_size,
                                            ff_dim=out_size*2,
                                           dropout=fusion_dropout)
        if llm_size == '1B':
            self.align_m = nn.Linear(out_size,2048)
            self.align_t = nn.Linear(out_size,2048)
            self.align_g = nn.Linear(out_size,2048)
            self.align_f = nn.Linear(out_size,2048)
        elif llm_size == '3B':
            self.align_m = nn.Linear(out_size,3072)
            self.align_t = nn.Linear(out_size,3072)
            self.align_g = nn.Linear(out_size,3072)
            self.align_f = nn.Linear(out_size,3072)
        elif llm_size == '8B':
            self.align_m = nn.Linear(out_size,4096)
            self.align_t = nn.Linear(out_size,4096)
            self.align_g = nn.Linear(out_size,4096)
            self.align_f = nn.Linear(out_size,4096)
        if is_linear_init:
            self.linear_init()
        
    def init_llm_model(self,llm_path,freeze_param):
        # LLM 模型初始化
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='right')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        new_tokens = ["<metadata>","<text>","<graph>","<fusion>"]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, torch_dtype=torch.bfloat16)

        llm_model.resize_token_embeddings(len(tokenizer))
        llm_model.config.pad_token_id = tokenizer.pad_token_id

        pretoken_model_train=PreToken_Train(tokenizer)

        if freeze_param:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
        for param in llm_model.score.parameters():
            param.requires_grad = True
        
        return llm_model,pretoken_model_train
    
    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        # init_weights(self.mapping_m)
        # init_weights(self.mapping_t)
        # init_weights(self.mapping_g)
        # init_weights(self.mapping_f)
        init_weights(self.align_m)
        init_weights(self.align_t)
        init_weights(self.align_g)
        init_weights(self.align_f)
    
    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,prompt,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        meta_embed_aligned = self.align_m(meta_embed)
        text_embed_aligned = self.align_t(text_embed)
        graph_embed_aligned = self.align_g(graph_embed)
        fusion_embed_aligned = self.align_f(fusion_embed)

        instruction_tokens = self.pretoken_model_train(prompt,64)

        # 确保所有张量在同一设备上
        instruction_tokens = instruction_tokens.to(self.device)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            
        with torch.amp.autocast("cuda",dtype=torch.float16):

            # graph_inputs_llm = fusion_embed # [batch_size, 4096]
            meta_inputs_llm = meta_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            text_inputs_llm = text_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            graph_inputs_llm = graph_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            fusion_inputs_llm = fusion_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]

            # llama_word_embeddings = self.word_embeddings.unsqueeze(0) # [1,32001,4096]
            # llama_word_embeddings = llama_word_embeddings.repeat(
            # graph_inputs_llm.size(0), 1, 1).to(self.device)# 沿第一个维度复制 llama_word_embeddings,维度变为[batch_size,32001,4096]

            # cat_embedding_m = self.get_cat_embedding(meta_inputs_llm,llama_word_embeddings,"m")
            # cat_embedding_t = self.get_cat_embedding(text_inputs_llm,llama_word_embeddings,"t")
            # cat_embedding_g = self.get_cat_embedding(graph_inputs_llm,llama_word_embeddings,"g")
            # cat_embedding_f = self.get_cat_embedding(fusion_inputs_llm,llama_word_embeddings,"f")

            inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,graph_inputs_llm,fusion_inputs_llm)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=instruction_tokens.attention_mask,
                    return_dict=True,
                )
            logits = outputs.logits

        return logits

    def get_cat_embedding(self,modality_embeddings,llama_word_embeddings,modality):
        cat_embedding = torch.cat([modality_embeddings, llama_word_embeddings], dim=1)# [batch_size,32002,4096]
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()# [batch_size,4096,32002] # .contiguous()：确保张量在内存中的存储是连续的，避免可能的数据存储问题
        if modality=='m':
            cat_embedding = self.mapping_m(cat_embedding)# [batch_size,4096,1] # self.mapping是将32002->1
        elif modality=='t':
            cat_embedding = self.mapping_t(cat_embedding)
        elif modality=='g':
            cat_embedding = self.mapping_g(cat_embedding)
        elif modality=='f':
            cat_embedding = self.mapping_f(cat_embedding)
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

        return cat_embedding

    def concat_instruction_modality_embeds(self,instruction_embeds,embedding_m,embedding_t,embedding_g,embedding_f):
        # <metadata>:33
        # <text>:38
        # <graph>:43
        # <fusion>:49
        front_embeds = instruction_embeds[:,:33,:]
        # embedding_m
        mid_embeds1 = instruction_embeds[:,34:38,:]
        # embedding_t
        mid_embeds2 = instruction_embeds[:,39:43,:]
        # embedding_g
        mid_embeds3 = instruction_embeds[:,44:49,:]
        # embedding_f
        back_embeds = instruction_embeds[:,50:,:]
        new_embeds = torch.cat([front_embeds,
                                embedding_m,
                                mid_embeds1,
                                embedding_t,
                                mid_embeds2,
                                embedding_g,
                                mid_embeds3,
                                embedding_f,
                                back_embeds], dim=1)

        return new_embeds
    
class MY_MODEL_TRM3(nn.Module):
    def __init__(self,
                 llm_path,
                 llm_freeze_param,
                 out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 llm_size='1B',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=out_size,
                                     dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,hidden_size=metadata_hidden_size,output_size=out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)
        

        # self.llm_model,self.pretoken_model_train,self.word_embeddings,self.vocab_size = self.init_llm_model(llm_path,llm_freeze_param)

        self.llm_model,self.pretoken_model_train = self.init_llm_model(llm_path,llm_freeze_param)

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight # [32001,4096] # 32002
        # self.vocab_size = self.word_embeddings.shape[0] # 32002

        # self.mapping_m = nn.Linear(self.vocab_size + 1, 1)# 将词汇表从n+1维映射到1维 图2(c)步骤
        # self.mapping_t = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_g = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_f = nn.Linear(self.vocab_size + 1, 1)

        self.fusion = ModalityAttentionFusion_mean_pool(num_modalities=3,
                                            embed_dim=out_size,
                                            ff_dim=out_size*2,
                                           dropout=fusion_dropout)
        if llm_size == '1B':
            self.align_m = nn.Linear(out_size,2048)
            self.align_t = nn.Linear(out_size,2048)
            self.align_g = nn.Linear(out_size,2048)
            self.align_f = nn.Linear(out_size,2048)
        elif llm_size == '3B':
            self.align_m = nn.Linear(out_size,3072)
            self.align_t = nn.Linear(out_size,3072)
            self.align_g = nn.Linear(out_size,3072)
            self.align_f = nn.Linear(out_size,3072)
        elif llm_size == '8B':
            self.align_m = nn.Linear(out_size,4096)
            self.align_t = nn.Linear(out_size,4096)
            self.align_g = nn.Linear(out_size,4096)
            self.align_f = nn.Linear(out_size,4096)
        if is_linear_init:
            self.linear_init()
        
    def init_llm_model(self,llm_path,freeze_param):
        # LLM 模型初始化
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='right')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        new_tokens = ["<metadata>","<text>","<graph>","<fusion>"]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, torch_dtype=torch.bfloat16)

        llm_model.resize_token_embeddings(len(tokenizer))
        llm_model.config.pad_token_id = tokenizer.pad_token_id

        pretoken_model_train=PreToken_Train(tokenizer)

        if freeze_param:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
        for param in llm_model.score.parameters():
            param.requires_grad = True
        
        return llm_model,pretoken_model_train
    
    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        # init_weights(self.mapping_m)
        # init_weights(self.mapping_t)
        # init_weights(self.mapping_g)
        # init_weights(self.mapping_f)
        init_weights(self.align_m)
        init_weights(self.align_t)
        init_weights(self.align_g)
        init_weights(self.align_f)
    
    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,prompt,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        meta_embed_aligned = self.align_m(meta_embed)
        text_embed_aligned = self.align_t(text_embed)
        graph_embed_aligned = self.align_g(graph_embed)
        fusion_embed_aligned = self.align_f(fusion_embed)

        instruction_tokens = self.pretoken_model_train(prompt,64)

        # 确保所有张量在同一设备上
        instruction_tokens = instruction_tokens.to(self.device)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            
        with torch.amp.autocast("cuda",dtype=torch.float16):

            # graph_inputs_llm = fusion_embed # [batch_size, 4096]
            meta_inputs_llm = meta_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            text_inputs_llm = text_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            graph_inputs_llm = graph_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            fusion_inputs_llm = fusion_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]

            # llama_word_embeddings = self.word_embeddings.unsqueeze(0) # [1,32001,4096]
            # llama_word_embeddings = llama_word_embeddings.repeat(
            # graph_inputs_llm.size(0), 1, 1).to(self.device)# 沿第一个维度复制 llama_word_embeddings,维度变为[batch_size,32001,4096]

            # cat_embedding_m = self.get_cat_embedding(meta_inputs_llm,llama_word_embeddings,"m")
            # cat_embedding_t = self.get_cat_embedding(text_inputs_llm,llama_word_embeddings,"t")
            # cat_embedding_g = self.get_cat_embedding(graph_inputs_llm,llama_word_embeddings,"g")
            # cat_embedding_f = self.get_cat_embedding(fusion_inputs_llm,llama_word_embeddings,"f")

            inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,graph_inputs_llm,fusion_inputs_llm)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=instruction_tokens.attention_mask,
                    return_dict=True,
                )
            logits = outputs.logits

        return logits

    def get_cat_embedding(self,modality_embeddings,llama_word_embeddings,modality):
        cat_embedding = torch.cat([modality_embeddings, llama_word_embeddings], dim=1)# [batch_size,32002,4096]
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()# [batch_size,4096,32002] # .contiguous()：确保张量在内存中的存储是连续的，避免可能的数据存储问题
        if modality=='m':
            cat_embedding = self.mapping_m(cat_embedding)# [batch_size,4096,1] # self.mapping是将32002->1
        elif modality=='t':
            cat_embedding = self.mapping_t(cat_embedding)
        elif modality=='g':
            cat_embedding = self.mapping_g(cat_embedding)
        elif modality=='f':
            cat_embedding = self.mapping_f(cat_embedding)
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

        return cat_embedding

    def concat_instruction_modality_embeds(self,instruction_embeds,embedding_m,embedding_t,embedding_g,embedding_f):
        # <metadata>:33
        # <text>:38
        # <graph>:43
        # <fusion>:49
        front_embeds = instruction_embeds[:,:33,:]
        # embedding_m
        mid_embeds1 = instruction_embeds[:,34:38,:]
        # embedding_t
        mid_embeds2 = instruction_embeds[:,39:43,:]
        # embedding_g
        mid_embeds3 = instruction_embeds[:,44:49,:]
        # embedding_f
        back_embeds = instruction_embeds[:,50:,:]
        new_embeds = torch.cat([front_embeds,
                                embedding_m,
                                mid_embeds1,
                                embedding_t,
                                mid_embeds2,
                                embedding_g,
                                mid_embeds3,
                                embedding_f,
                                back_embeds], dim=1)

        return new_embeds
    
class MY_MODEL_TRM4(nn.Module):
    def __init__(self,
                 llm_path,
                 llm_freeze_param,
                 out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 llm_size='1B',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=out_size,
                                     dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,hidden_size=metadata_hidden_size,output_size=out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)
        

        # self.llm_model,self.pretoken_model_train,self.word_embeddings,self.vocab_size = self.init_llm_model(llm_path,llm_freeze_param)

        self.llm_model,self.pretoken_model_train = self.init_llm_model(llm_path,llm_freeze_param)

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight # [32001,4096] # 32002
        # self.vocab_size = self.word_embeddings.shape[0] # 32002

        # self.mapping_m = nn.Linear(self.vocab_size + 1, 1)# 将词汇表从n+1维映射到1维 图2(c)步骤
        # self.mapping_t = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_g = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_f = nn.Linear(self.vocab_size + 1, 1)

        self.fusion = ModalityAttentionFusion_max_pool(num_modalities=3,
                                            embed_dim=out_size,
                                            ff_dim=out_size*2,
                                           dropout=fusion_dropout)
        if llm_size == '1B':
            self.align_m = nn.Linear(out_size,2048)
            self.align_t = nn.Linear(out_size,2048)
            self.align_g = nn.Linear(out_size,2048)
            self.align_f = nn.Linear(out_size,2048)
        elif llm_size == '3B':
            self.align_m = nn.Linear(out_size,3072)
            self.align_t = nn.Linear(out_size,3072)
            self.align_g = nn.Linear(out_size,3072)
            self.align_f = nn.Linear(out_size,3072)
        elif llm_size == '8B':
            self.align_m = nn.Linear(out_size,4096)
            self.align_t = nn.Linear(out_size,4096)
            self.align_g = nn.Linear(out_size,4096)
            self.align_f = nn.Linear(out_size,4096)
        if is_linear_init:
            self.linear_init()
        
    def init_llm_model(self,llm_path,freeze_param):
        # LLM 模型初始化
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='right')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        new_tokens = ["<metadata>","<text>","<graph>","<fusion>"]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, torch_dtype=torch.bfloat16)

        llm_model.resize_token_embeddings(len(tokenizer))
        llm_model.config.pad_token_id = tokenizer.pad_token_id

        pretoken_model_train=PreToken_Train(tokenizer)

        if freeze_param:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
        for param in llm_model.score.parameters():
            param.requires_grad = True
        
        return llm_model,pretoken_model_train
    
    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        # init_weights(self.mapping_m)
        # init_weights(self.mapping_t)
        # init_weights(self.mapping_g)
        # init_weights(self.mapping_f)
        init_weights(self.align_m)
        init_weights(self.align_t)
        init_weights(self.align_g)
        init_weights(self.align_f)
    
    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,prompt,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        meta_embed_aligned = self.align_m(meta_embed)
        text_embed_aligned = self.align_t(text_embed)
        graph_embed_aligned = self.align_g(graph_embed)
        fusion_embed_aligned = self.align_f(fusion_embed)

        instruction_tokens = self.pretoken_model_train(prompt,64)

        # 确保所有张量在同一设备上
        instruction_tokens = instruction_tokens.to(self.device)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            
        with torch.amp.autocast("cuda",dtype=torch.float16):

            # graph_inputs_llm = fusion_embed # [batch_size, 4096]
            meta_inputs_llm = meta_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            text_inputs_llm = text_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            graph_inputs_llm = graph_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            fusion_inputs_llm = fusion_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]

            # llama_word_embeddings = self.word_embeddings.unsqueeze(0) # [1,32001,4096]
            # llama_word_embeddings = llama_word_embeddings.repeat(
            # graph_inputs_llm.size(0), 1, 1).to(self.device)# 沿第一个维度复制 llama_word_embeddings,维度变为[batch_size,32001,4096]

            # cat_embedding_m = self.get_cat_embedding(meta_inputs_llm,llama_word_embeddings,"m")
            # cat_embedding_t = self.get_cat_embedding(text_inputs_llm,llama_word_embeddings,"t")
            # cat_embedding_g = self.get_cat_embedding(graph_inputs_llm,llama_word_embeddings,"g")
            # cat_embedding_f = self.get_cat_embedding(fusion_inputs_llm,llama_word_embeddings,"f")

            inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,graph_inputs_llm,fusion_inputs_llm)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=instruction_tokens.attention_mask,
                    return_dict=True,
                )
            logits = outputs.logits

        return logits

    def get_cat_embedding(self,modality_embeddings,llama_word_embeddings,modality):
        cat_embedding = torch.cat([modality_embeddings, llama_word_embeddings], dim=1)# [batch_size,32002,4096]
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()# [batch_size,4096,32002] # .contiguous()：确保张量在内存中的存储是连续的，避免可能的数据存储问题
        if modality=='m':
            cat_embedding = self.mapping_m(cat_embedding)# [batch_size,4096,1] # self.mapping是将32002->1
        elif modality=='t':
            cat_embedding = self.mapping_t(cat_embedding)
        elif modality=='g':
            cat_embedding = self.mapping_g(cat_embedding)
        elif modality=='f':
            cat_embedding = self.mapping_f(cat_embedding)
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

        return cat_embedding

    def concat_instruction_modality_embeds(self,instruction_embeds,embedding_m,embedding_t,embedding_g,embedding_f):
        # <metadata>:33
        # <text>:38
        # <graph>:43
        # <fusion>:49
        front_embeds = instruction_embeds[:,:33,:]
        # embedding_m
        mid_embeds1 = instruction_embeds[:,34:38,:]
        # embedding_t
        mid_embeds2 = instruction_embeds[:,39:43,:]
        # embedding_g
        mid_embeds3 = instruction_embeds[:,44:49,:]
        # embedding_f
        back_embeds = instruction_embeds[:,50:,:]
        new_embeds = torch.cat([front_embeds,
                                embedding_m,
                                mid_embeds1,
                                embedding_t,
                                mid_embeds2,
                                embedding_g,
                                mid_embeds3,
                                embedding_f,
                                back_embeds], dim=1)

        return new_embeds

class MY_MODEL_TRM5(nn.Module):
    def __init__(self,
                 llm_path,
                 llm_freeze_param,
                 out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 llm_size='1B',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=out_size,
                                     dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,hidden_size=metadata_hidden_size,output_size=out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)
        

        # self.llm_model,self.pretoken_model_train,self.word_embeddings,self.vocab_size = self.init_llm_model(llm_path,llm_freeze_param)

        self.llm_model,self.pretoken_model_train = self.init_llm_model(llm_path,llm_freeze_param)

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight # [32001,4096] # 32002
        # self.vocab_size = self.word_embeddings.shape[0] # 32002

        # self.mapping_m = nn.Linear(self.vocab_size + 1, 1)# 将词汇表从n+1维映射到1维 图2(c)步骤
        # self.mapping_t = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_g = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_f = nn.Linear(self.vocab_size + 1, 1)

        self.fusion = ModalityAttentionFusion_min_pool(num_modalities=3,
                                            embed_dim=out_size,
                                            ff_dim=out_size*2,
                                           dropout=fusion_dropout)
        if llm_size == '1B':
            self.align_m = nn.Linear(out_size,2048)
            self.align_t = nn.Linear(out_size,2048)
            self.align_g = nn.Linear(out_size,2048)
            self.align_f = nn.Linear(out_size,2048)
        elif llm_size == '3B':
            self.align_m = nn.Linear(out_size,3072)
            self.align_t = nn.Linear(out_size,3072)
            self.align_g = nn.Linear(out_size,3072)
            self.align_f = nn.Linear(out_size,3072)
        elif llm_size == '8B':
            self.align_m = nn.Linear(out_size,4096)
            self.align_t = nn.Linear(out_size,4096)
            self.align_g = nn.Linear(out_size,4096)
            self.align_f = nn.Linear(out_size,4096)
        if is_linear_init:
            self.linear_init()
        
    def init_llm_model(self,llm_path,freeze_param):
        # LLM 模型初始化
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='right')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        new_tokens = ["<metadata>","<text>","<graph>","<fusion>"]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, torch_dtype=torch.bfloat16)

        llm_model.resize_token_embeddings(len(tokenizer))
        llm_model.config.pad_token_id = tokenizer.pad_token_id

        pretoken_model_train=PreToken_Train(tokenizer)

        if freeze_param:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
        for param in llm_model.score.parameters():
            param.requires_grad = True
        
        return llm_model,pretoken_model_train
    
    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        # init_weights(self.mapping_m)
        # init_weights(self.mapping_t)
        # init_weights(self.mapping_g)
        # init_weights(self.mapping_f)
        init_weights(self.align_m)
        init_weights(self.align_t)
        init_weights(self.align_g)
        init_weights(self.align_f)
    
    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,prompt,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        meta_embed_aligned = self.align_m(meta_embed)
        text_embed_aligned = self.align_t(text_embed)
        graph_embed_aligned = self.align_g(graph_embed)
        fusion_embed_aligned = self.align_f(fusion_embed)

        instruction_tokens = self.pretoken_model_train(prompt,64)

        # 确保所有张量在同一设备上
        instruction_tokens = instruction_tokens.to(self.device)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            
        with torch.amp.autocast("cuda",dtype=torch.float16):

            # graph_inputs_llm = fusion_embed # [batch_size, 4096]
            meta_inputs_llm = meta_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            text_inputs_llm = text_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            graph_inputs_llm = graph_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            fusion_inputs_llm = fusion_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]

            # llama_word_embeddings = self.word_embeddings.unsqueeze(0) # [1,32001,4096]
            # llama_word_embeddings = llama_word_embeddings.repeat(
            # graph_inputs_llm.size(0), 1, 1).to(self.device)# 沿第一个维度复制 llama_word_embeddings,维度变为[batch_size,32001,4096]

            # cat_embedding_m = self.get_cat_embedding(meta_inputs_llm,llama_word_embeddings,"m")
            # cat_embedding_t = self.get_cat_embedding(text_inputs_llm,llama_word_embeddings,"t")
            # cat_embedding_g = self.get_cat_embedding(graph_inputs_llm,llama_word_embeddings,"g")
            # cat_embedding_f = self.get_cat_embedding(fusion_inputs_llm,llama_word_embeddings,"f")

            inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,graph_inputs_llm,fusion_inputs_llm)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=instruction_tokens.attention_mask,
                    return_dict=True,
                )
            logits = outputs.logits

        return logits

    def get_cat_embedding(self,modality_embeddings,llama_word_embeddings,modality):
        cat_embedding = torch.cat([modality_embeddings, llama_word_embeddings], dim=1)# [batch_size,32002,4096]
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()# [batch_size,4096,32002] # .contiguous()：确保张量在内存中的存储是连续的，避免可能的数据存储问题
        if modality=='m':
            cat_embedding = self.mapping_m(cat_embedding)# [batch_size,4096,1] # self.mapping是将32002->1
        elif modality=='t':
            cat_embedding = self.mapping_t(cat_embedding)
        elif modality=='g':
            cat_embedding = self.mapping_g(cat_embedding)
        elif modality=='f':
            cat_embedding = self.mapping_f(cat_embedding)
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

        return cat_embedding

    def concat_instruction_modality_embeds(self,instruction_embeds,embedding_m,embedding_t,embedding_g,embedding_f):
        # <metadata>:33
        # <text>:38
        # <graph>:43
        # <fusion>:49
        front_embeds = instruction_embeds[:,:33,:]
        # embedding_m
        mid_embeds1 = instruction_embeds[:,34:38,:]
        # embedding_t
        mid_embeds2 = instruction_embeds[:,39:43,:]
        # embedding_g
        mid_embeds3 = instruction_embeds[:,44:49,:]
        # embedding_f
        back_embeds = instruction_embeds[:,50:,:]
        new_embeds = torch.cat([front_embeds,
                                embedding_m,
                                mid_embeds1,
                                embedding_t,
                                mid_embeds2,
                                embedding_g,
                                mid_embeds3,
                                embedding_f,
                                back_embeds], dim=1)

        return new_embeds
    
class MY_MODEL_without_graph(nn.Module):
    def __init__(self,
                 llm_path,
                 llm_freeze_param,
                 out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 llm_size='1B',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        # self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
        #                              cat_prop_size=in_cat_prop_size,
        #                              embedding_dimension=out_size,
        #                              dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,hidden_size=metadata_hidden_size,output_size=out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)
        

        # self.llm_model,self.pretoken_model_train,self.word_embeddings,self.vocab_size = self.init_llm_model(llm_path,llm_freeze_param)

        self.llm_model,self.pretoken_model_train = self.init_llm_model(llm_path,llm_freeze_param)

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight # [32001,4096] # 32002
        # self.vocab_size = self.word_embeddings.shape[0] # 32002

        # self.mapping_m = nn.Linear(self.vocab_size + 1, 1)# 将词汇表从n+1维映射到1维 图2(c)步骤
        # self.mapping_t = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_g = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_f = nn.Linear(self.vocab_size + 1, 1)

        self.fusion = ModalityAttentionFusion(num_modalities=2,
                                            embed_dim=out_size,
                                            ff_dim=out_size*2,
                                           dropout=fusion_dropout)
        if llm_size == '1B':
            self.align_m = nn.Linear(out_size,2048)
            self.align_t = nn.Linear(out_size,2048)
            self.align_g = nn.Linear(out_size,2048)
            self.align_f = nn.Linear(out_size,2048)
        elif llm_size == '3B':
            self.align_m = nn.Linear(out_size,3072)
            self.align_t = nn.Linear(out_size,3072)
            self.align_g = nn.Linear(out_size,3072)
            self.align_f = nn.Linear(out_size,3072)
        elif llm_size == '8B':
            self.align_m = nn.Linear(out_size,4096)
            self.align_t = nn.Linear(out_size,4096)
            self.align_g = nn.Linear(out_size,4096)
            self.align_f = nn.Linear(out_size,4096)
        if is_linear_init:
            self.linear_init()
        
    def init_llm_model(self,llm_path,freeze_param):
        # LLM 模型初始化
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='right')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        # new_tokens = ["<metadata>","<text>","<graph>","<fusion>"]
        new_tokens = ["<metadata>","<text>","<fusion>"]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, torch_dtype=torch.bfloat16)

        llm_model.resize_token_embeddings(len(tokenizer))
        llm_model.config.pad_token_id = tokenizer.pad_token_id

        pretoken_model_train=PreToken_Train(tokenizer)

        if freeze_param:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
        for param in llm_model.score.parameters():
            param.requires_grad = True
        
        return llm_model,pretoken_model_train
    
    def linear_init(self):
        # self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        # init_weights(self.mapping_m)
        # init_weights(self.mapping_t)
        # init_weights(self.mapping_g)
        # init_weights(self.mapping_f)
        init_weights(self.align_m)
        init_weights(self.align_t)
        # init_weights(self.align_g)
        init_weights(self.align_f)
    
    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,prompt,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        # graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        # fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])
        fusion_embed = self.fusion([meta_embed,text_embed])

        meta_embed_aligned = self.align_m(meta_embed)
        text_embed_aligned = self.align_t(text_embed)
        # graph_embed_aligned = self.align_g(graph_embed)
        fusion_embed_aligned = self.align_f(fusion_embed)

        instruction_tokens = self.pretoken_model_train(prompt,64)

        # 确保所有张量在同一设备上
        instruction_tokens = instruction_tokens.to(self.device)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            
        with torch.amp.autocast("cuda",dtype=torch.float16):

            # graph_inputs_llm = fusion_embed # [batch_size, 4096]
            meta_inputs_llm = meta_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            text_inputs_llm = text_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            # graph_inputs_llm = graph_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            fusion_inputs_llm = fusion_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]

            # llama_word_embeddings = self.word_embeddings.unsqueeze(0) # [1,32001,4096]
            # llama_word_embeddings = llama_word_embeddings.repeat(
            # graph_inputs_llm.size(0), 1, 1).to(self.device)# 沿第一个维度复制 llama_word_embeddings,维度变为[batch_size,32001,4096]

            # cat_embedding_m = self.get_cat_embedding(meta_inputs_llm,llama_word_embeddings,"m")
            # cat_embedding_t = self.get_cat_embedding(text_inputs_llm,llama_word_embeddings,"t")
            # cat_embedding_g = self.get_cat_embedding(graph_inputs_llm,llama_word_embeddings,"g")
            # cat_embedding_f = self.get_cat_embedding(fusion_inputs_llm,llama_word_embeddings,"f")

            # inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,graph_inputs_llm,fusion_inputs_llm)
            inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,fusion_inputs_llm)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=instruction_tokens.attention_mask,
                    return_dict=True,
                )
            logits = outputs.logits

        return logits

    def get_cat_embedding(self,modality_embeddings,llama_word_embeddings,modality):
        cat_embedding = torch.cat([modality_embeddings, llama_word_embeddings], dim=1)# [batch_size,32002,4096]
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()# [batch_size,4096,32002] # .contiguous()：确保张量在内存中的存储是连续的，避免可能的数据存储问题
        if modality=='m':
            cat_embedding = self.mapping_m(cat_embedding)# [batch_size,4096,1] # self.mapping是将32002->1
        elif modality=='t':
            cat_embedding = self.mapping_t(cat_embedding)
        # elif modality=='g':
        #     cat_embedding = self.mapping_g(cat_embedding)
        elif modality=='f':
            cat_embedding = self.mapping_f(cat_embedding)
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

        return cat_embedding

    def concat_instruction_modality_embeds(self,instruction_embeds,embedding_m,embedding_t,embedding_f):
        # <metadata>:30
        # <text>:35
        # <fusion>:41
        front_embeds = instruction_embeds[:,:30,:]  # 0-29
        # embedding_m                               # 30
        mid_embeds1 = instruction_embeds[:,31:35,:] # 31-34
        # embedding_t                               # 35
        mid_embeds2 = instruction_embeds[:,36:41,:] # 36-40
        # embedding_f                               # 41
        back_embeds = instruction_embeds[:,42:,:]   # 42-
        new_embeds = torch.cat([front_embeds,
                                embedding_m,
                                mid_embeds1,
                                embedding_t,
                                mid_embeds2,
                                embedding_f,
                                back_embeds], dim=1)

        return new_embeds
    
class MY_MODEL_without_metadata(nn.Module):
    def __init__(self,
                 llm_path,
                 llm_freeze_param,
                 out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 llm_size='1B',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=out_size,
                                     dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        # self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
        #                                       dim2=in_cat_prop_size,hidden_size=metadata_hidden_size,output_size=out_size,
        #                                       fusion_type=fusion_type, 
        #                                       activation='gelu',
        #                                       dropout=0.1)
        

        # self.llm_model,self.pretoken_model_train,self.word_embeddings,self.vocab_size = self.init_llm_model(llm_path,llm_freeze_param)

        self.llm_model,self.pretoken_model_train = self.init_llm_model(llm_path,llm_freeze_param)

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight # [32001,4096] # 32002
        # self.vocab_size = self.word_embeddings.shape[0] # 32002

        # self.mapping_m = nn.Linear(self.vocab_size + 1, 1)# 将词汇表从n+1维映射到1维 图2(c)步骤
        # self.mapping_t = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_g = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_f = nn.Linear(self.vocab_size + 1, 1)

        self.fusion = ModalityAttentionFusion(num_modalities=2,
                                            embed_dim=out_size,
                                            ff_dim=out_size*2,
                                           dropout=fusion_dropout)
        if llm_size == '1B':
            self.align_m = nn.Linear(out_size,2048)
            self.align_t = nn.Linear(out_size,2048)
            self.align_g = nn.Linear(out_size,2048)
            self.align_f = nn.Linear(out_size,2048)
        elif llm_size == '3B':
            self.align_m = nn.Linear(out_size,3072)
            self.align_t = nn.Linear(out_size,3072)
            self.align_g = nn.Linear(out_size,3072)
            self.align_f = nn.Linear(out_size,3072)
        elif llm_size == '8B':
            self.align_m = nn.Linear(out_size,4096)
            self.align_t = nn.Linear(out_size,4096)
            self.align_g = nn.Linear(out_size,4096)
            self.align_f = nn.Linear(out_size,4096)
        if is_linear_init:
            self.linear_init()
        
    def init_llm_model(self,llm_path,freeze_param):
        # LLM 模型初始化
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='right')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        # new_tokens = ["<metadata>","<text>","<graph>","<fusion>"]
        new_tokens = ["<text>","<graph>","<fusion>"]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, torch_dtype=torch.bfloat16)

        llm_model.resize_token_embeddings(len(tokenizer))
        llm_model.config.pad_token_id = tokenizer.pad_token_id

        pretoken_model_train=PreToken_Train(tokenizer)

        if freeze_param:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
        for param in llm_model.score.parameters():
            param.requires_grad = True
        
        return llm_model,pretoken_model_train
    
    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        # self.metadata_encoder.apply(init_weights)
        # init_weights(self.mapping_m)
        # init_weights(self.mapping_t)
        # init_weights(self.mapping_g)
        # init_weights(self.mapping_f)
        # init_weights(self.align_m)
        init_weights(self.align_t)
        init_weights(self.align_g)
        init_weights(self.align_f)
    
    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,prompt,idx):
        # meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        # fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])
        fusion_embed = self.fusion([text_embed,graph_embed])

        # meta_embed_aligned = self.align_m(meta_embed)
        text_embed_aligned = self.align_t(text_embed)
        graph_embed_aligned = self.align_g(graph_embed)
        fusion_embed_aligned = self.align_f(fusion_embed)

        instruction_tokens = self.pretoken_model_train(prompt,64)

        # 确保所有张量在同一设备上
        instruction_tokens = instruction_tokens.to(self.device)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            
        with torch.amp.autocast("cuda",dtype=torch.float16):

            # graph_inputs_llm = fusion_embed # [batch_size, 4096]
            # meta_inputs_llm = meta_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            text_inputs_llm = text_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            graph_inputs_llm = graph_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            fusion_inputs_llm = fusion_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]

            # llama_word_embeddings = self.word_embeddings.unsqueeze(0) # [1,32001,4096]
            # llama_word_embeddings = llama_word_embeddings.repeat(
            # graph_inputs_llm.size(0), 1, 1).to(self.device)# 沿第一个维度复制 llama_word_embeddings,维度变为[batch_size,32001,4096]

            # cat_embedding_m = self.get_cat_embedding(meta_inputs_llm,llama_word_embeddings,"m")
            # cat_embedding_t = self.get_cat_embedding(text_inputs_llm,llama_word_embeddings,"t")
            # cat_embedding_g = self.get_cat_embedding(graph_inputs_llm,llama_word_embeddings,"g")
            # cat_embedding_f = self.get_cat_embedding(fusion_inputs_llm,llama_word_embeddings,"f")

            # inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,graph_inputs_llm,fusion_inputs_llm)
            inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,text_inputs_llm,graph_inputs_llm,fusion_inputs_llm)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=instruction_tokens.attention_mask,
                    return_dict=True,
                )
            logits = outputs.logits

        return logits

    def get_cat_embedding(self,modality_embeddings,llama_word_embeddings,modality):
        cat_embedding = torch.cat([modality_embeddings, llama_word_embeddings], dim=1)# [batch_size,32002,4096]
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()# [batch_size,4096,32002] # .contiguous()：确保张量在内存中的存储是连续的，避免可能的数据存储问题
        if modality=='m':
            cat_embedding = self.mapping_m(cat_embedding)# [batch_size,4096,1] # self.mapping是将32002->1
        elif modality=='t':
            cat_embedding = self.mapping_t(cat_embedding)
        # elif modality=='g':
        #     cat_embedding = self.mapping_g(cat_embedding)
        elif modality=='f':
            cat_embedding = self.mapping_f(cat_embedding)
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

        return cat_embedding

    def concat_instruction_modality_embeds(self,instruction_embeds,embedding_t,embedding_g,embedding_f):
        # <text>:33
        # <graph>:38
        # <fusion>:44
        front_embeds = instruction_embeds[:,:33,:]  # 0-32
        # embedding_t                               # 33
        mid_embeds1 = instruction_embeds[:,34:38,:] # 34-37
        # embedding_g                               # 38
        mid_embeds2 = instruction_embeds[:,39:44,:] # 39-43
        # embedding_f                               # 44
        back_embeds = instruction_embeds[:,45:,:]   # 45-
        new_embeds = torch.cat([front_embeds,
                                embedding_t,
                                mid_embeds1,
                                embedding_g,
                                mid_embeds2,
                                embedding_f,
                                back_embeds], dim=1)

        return new_embeds
    
class MY_MODEL_without_text(nn.Module):
    def __init__(self,
                 llm_path,
                 llm_freeze_param,
                 out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 llm_size='1B',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=out_size,
                                     dropout=gnn_dropout)
        # self.text_encoder = FeatureFusion2(dim1=768,
        #                                     dim2=768,
        #                                     dim3=768,
        #                                     hidden_size=text_hidden_size,
        #                                     output_size=out_size,
        #                                     fusion_type=fusion_type, 
        #                                     activation='gelu', 
        #                                     dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,hidden_size=metadata_hidden_size,output_size=out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)
        

        # self.llm_model,self.pretoken_model_train,self.word_embeddings,self.vocab_size = self.init_llm_model(llm_path,llm_freeze_param)

        self.llm_model,self.pretoken_model_train = self.init_llm_model(llm_path,llm_freeze_param)

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight # [32001,4096] # 32002
        # self.vocab_size = self.word_embeddings.shape[0] # 32002

        # self.mapping_m = nn.Linear(self.vocab_size + 1, 1)# 将词汇表从n+1维映射到1维 图2(c)步骤
        # self.mapping_t = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_g = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_f = nn.Linear(self.vocab_size + 1, 1)

        self.fusion = ModalityAttentionFusion(num_modalities=2,
                                            embed_dim=out_size,
                                            ff_dim=out_size*2,
                                           dropout=fusion_dropout)
        if llm_size == '1B':
            self.align_m = nn.Linear(out_size,2048)
            self.align_t = nn.Linear(out_size,2048)
            self.align_g = nn.Linear(out_size,2048)
            self.align_f = nn.Linear(out_size,2048)
        elif llm_size == '3B':
            self.align_m = nn.Linear(out_size,3072)
            self.align_t = nn.Linear(out_size,3072)
            self.align_g = nn.Linear(out_size,3072)
            self.align_f = nn.Linear(out_size,3072)
        elif llm_size == '8B':
            self.align_m = nn.Linear(out_size,4096)
            self.align_t = nn.Linear(out_size,4096)
            self.align_g = nn.Linear(out_size,4096)
            self.align_f = nn.Linear(out_size,4096)
        if is_linear_init:
            self.linear_init()
        
    def init_llm_model(self,llm_path,freeze_param):
        # LLM 模型初始化
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='right')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        # new_tokens = ["<metadata>","<text>","<graph>","<fusion>"]
        new_tokens = ["<metadata>","<graph>","<fusion>"]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, torch_dtype=torch.bfloat16)

        llm_model.resize_token_embeddings(len(tokenizer))
        llm_model.config.pad_token_id = tokenizer.pad_token_id

        pretoken_model_train=PreToken_Train(tokenizer)

        if freeze_param:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
        for param in llm_model.score.parameters():
            param.requires_grad = True
        
        return llm_model,pretoken_model_train
    
    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        # self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        # init_weights(self.mapping_m)
        # init_weights(self.mapping_t)
        # init_weights(self.mapping_g)
        # init_weights(self.mapping_f)
        init_weights(self.align_m)
        # init_weights(self.align_t)
        init_weights(self.align_g)
        init_weights(self.align_f)
    
    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,prompt,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        # text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        # fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])
        fusion_embed = self.fusion([meta_embed,graph_embed])

        meta_embed_aligned = self.align_m(meta_embed)
        # text_embed_aligned = self.align_t(text_embed)
        graph_embed_aligned = self.align_g(graph_embed)
        fusion_embed_aligned = self.align_f(fusion_embed)

        instruction_tokens = self.pretoken_model_train(prompt,64)

        # 确保所有张量在同一设备上
        instruction_tokens = instruction_tokens.to(self.device)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            
        with torch.amp.autocast("cuda",dtype=torch.float16):

            # graph_inputs_llm = fusion_embed # [batch_size, 4096]
            meta_inputs_llm = meta_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            # text_inputs_llm = text_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            graph_inputs_llm = graph_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            fusion_inputs_llm = fusion_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]

            # llama_word_embeddings = self.word_embeddings.unsqueeze(0) # [1,32001,4096]
            # llama_word_embeddings = llama_word_embeddings.repeat(
            # graph_inputs_llm.size(0), 1, 1).to(self.device)# 沿第一个维度复制 llama_word_embeddings,维度变为[batch_size,32001,4096]

            # cat_embedding_m = self.get_cat_embedding(meta_inputs_llm,llama_word_embeddings,"m")
            # cat_embedding_t = self.get_cat_embedding(text_inputs_llm,llama_word_embeddings,"t")
            # cat_embedding_g = self.get_cat_embedding(graph_inputs_llm,llama_word_embeddings,"g")
            # cat_embedding_f = self.get_cat_embedding(fusion_inputs_llm,llama_word_embeddings,"f")

            # inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,graph_inputs_llm,fusion_inputs_llm)
            inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,graph_inputs_llm,fusion_inputs_llm)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=instruction_tokens.attention_mask,
                    return_dict=True,
                )
            logits = outputs.logits

        return logits

    def get_cat_embedding(self,modality_embeddings,llama_word_embeddings,modality):
        cat_embedding = torch.cat([modality_embeddings, llama_word_embeddings], dim=1)# [batch_size,32002,4096]
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()# [batch_size,4096,32002] # .contiguous()：确保张量在内存中的存储是连续的，避免可能的数据存储问题
        if modality=='m':
            cat_embedding = self.mapping_m(cat_embedding)# [batch_size,4096,1] # self.mapping是将32002->1
        elif modality=='t':
            cat_embedding = self.mapping_t(cat_embedding)
        # elif modality=='g':
        #     cat_embedding = self.mapping_g(cat_embedding)
        elif modality=='f':
            cat_embedding = self.mapping_f(cat_embedding)
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

        return cat_embedding

    def concat_instruction_modality_embeds(self,instruction_embeds,embedding_m,embedding_g,embedding_f):
        # <metadata>:31
        # <graph>:36
        # <fusion>:42
        front_embeds = instruction_embeds[:,:31,:]  # 0-30
        # embedding_t                               # 31
        mid_embeds1 = instruction_embeds[:,32:36,:] # 32-35
        # embedding_g                               # 36
        mid_embeds2 = instruction_embeds[:,37:42,:] # 37-41
        # embedding_f                               # 42
        back_embeds = instruction_embeds[:,43:,:]   # 43-
        new_embeds = torch.cat([front_embeds,
                                embedding_m,
                                mid_embeds1,
                                embedding_g,
                                mid_embeds2,
                                embedding_f,
                                back_embeds], dim=1)

        return new_embeds
    
class MY_MODEL_without_SA(nn.Module):
    def __init__(self,
                 llm_path,
                 llm_freeze_param,
                 out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 llm_size='1B',
                 fusion_type='concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(embedding_dimension=out_size,dropout=gnn_dropout)
        # self.graph_encoder = Graph_Encoder_RGT(embedding_dimension=out_size)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,hidden_size=metadata_hidden_size,output_size=out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)
        

        self.llm_model,self.pretoken_model_train = self.init_llm_model(llm_path,llm_freeze_param)

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight # [32001,4096] # 32002
        # self.vocab_size = self.word_embeddings.shape[0] # 32002

        # self.mapping_m = nn.Linear(self.vocab_size + 1, 1)# 将词汇表从n+1维映射到1维 图2(c)步骤
        # self.mapping_t = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_g = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_f = nn.Linear(self.vocab_size + 1, 1)

        self.fusion = ModalityAttentionFusion(num_modalities=3,
                                            embed_dim=out_size,
                                            ff_dim=out_size*2,
                                           dropout=fusion_dropout)
        if llm_size == '1B':
            self.align_m = nn.Linear(out_size,2048)
            self.align_t = nn.Linear(out_size,2048)
            self.align_g = nn.Linear(out_size,2048)
            self.align_f = nn.Linear(out_size,2048)
        elif llm_size == '3B':
            self.align_m = nn.Linear(out_size,3072)
            self.align_t = nn.Linear(out_size,3072)
            self.align_g = nn.Linear(out_size,3072)
            self.align_f = nn.Linear(out_size,3072)
        elif llm_size == '8B':
            self.align_m = nn.Linear(out_size,4096)
            self.align_t = nn.Linear(out_size,4096)
            self.align_g = nn.Linear(out_size,4096)
            self.align_f = nn.Linear(out_size,4096)
        if is_linear_init:
            self.linear_init()
        
    def init_llm_model(self,llm_path,freeze_param):
        # LLM 模型初始化
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='right')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        new_tokens = ["<metadata>","<text>","<graph>","<fusion>"]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, torch_dtype=torch.bfloat16)

        llm_model.resize_token_embeddings(len(tokenizer))
        llm_model.config.pad_token_id = tokenizer.pad_token_id

        pretoken_model_train=PreToken_Train(tokenizer)

        if freeze_param:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
        for param in llm_model.score.parameters():
            param.requires_grad = True
        
        return llm_model,pretoken_model_train
    
    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        # init_weights(self.mapping_m)
        # init_weights(self.mapping_t)
        # init_weights(self.mapping_g)
        # init_weights(self.mapping_f)
        init_weights(self.align_m)
        init_weights(self.align_t)
        init_weights(self.align_g)
        init_weights(self.align_f)
    
    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,prompt,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        meta_embed_aligned = self.align_m(meta_embed)
        text_embed_aligned = self.align_t(text_embed)
        graph_embed_aligned = self.align_g(graph_embed)
        fusion_embed_aligned = self.align_f(fusion_embed)

        instruction_tokens = self.pretoken_model_train(prompt,64)

        # 确保所有张量在同一设备上
        instruction_tokens = instruction_tokens.to(self.device)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            
        with torch.amp.autocast("cuda",dtype=torch.float16):

            # graph_inputs_llm = fusion_embed # [batch_size, 4096]
            meta_inputs_llm = meta_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            text_inputs_llm = text_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            graph_inputs_llm = graph_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            fusion_inputs_llm = fusion_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]

            # llama_word_embeddings = self.word_embeddings.unsqueeze(0) # [1,32001,4096]
            # llama_word_embeddings = llama_word_embeddings.repeat(
            # graph_inputs_llm.size(0), 1, 1).to(self.device)# 沿第一个维度复制 llama_word_embeddings,维度变为[batch_size,32001,4096]

            # cat_embedding_m = self.get_cat_embedding(meta_inputs_llm,llama_word_embeddings,"m")
            # cat_embedding_t = self.get_cat_embedding(text_inputs_llm,llama_word_embeddings,"t")
            # cat_embedding_g = self.get_cat_embedding(graph_inputs_llm,llama_word_embeddings,"g")
            # cat_embedding_f = self.get_cat_embedding(fusion_inputs_llm,llama_word_embeddings,"f")

            # inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,cat_embedding_m,cat_embedding_t,cat_embedding_g,cat_embedding_f)
            inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,graph_inputs_llm,fusion_inputs_llm)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=instruction_tokens.attention_mask,
                    return_dict=True,
                )
            logits = outputs.logits

        return logits

    def get_cat_embedding(self,modality_embeddings,llama_word_embeddings,modality):
        cat_embedding = torch.cat([modality_embeddings, llama_word_embeddings], dim=1)# [batch_size,32002,4096]
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()# [batch_size,4096,32002] # .contiguous()：确保张量在内存中的存储是连续的，避免可能的数据存储问题
        if modality=='m':
            cat_embedding = self.mapping_m(cat_embedding)# [batch_size,4096,1] # self.mapping是将32002->1
        elif modality=='t':
            cat_embedding = self.mapping_t(cat_embedding)
        elif modality=='g':
            cat_embedding = self.mapping_g(cat_embedding)
        elif modality=='f':
            cat_embedding = self.mapping_f(cat_embedding)
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

        return cat_embedding

    def concat_instruction_modality_embeds(self,instruction_embeds,embedding_m,embedding_t,embedding_g,embedding_f):
        # <metadata>:33
        # <text>:38
        # <graph>:43
        # <fusion>:49
        front_embeds = instruction_embeds[:,:33,:]
        # embedding_m
        mid_embeds1 = instruction_embeds[:,34:38,:]
        # embedding_t
        mid_embeds2 = instruction_embeds[:,39:43,:]
        # embedding_g
        mid_embeds3 = instruction_embeds[:,44:49,:]
        # embedding_f
        back_embeds = instruction_embeds[:,50:,:]
        new_embeds = torch.cat([front_embeds,
                                embedding_m,
                                mid_embeds1,
                                embedding_t,
                                mid_embeds2,
                                embedding_g,
                                mid_embeds3,
                                embedding_f,
                                back_embeds], dim=1)

        return new_embeds

class TransformerAggregator(nn.Module):
    def __init__(
        self,
        input_dim=32,          # 每个模态输入维度，例如你的 hm/ht/hg/hf 是 32
        hidden_dim=256,        # Transformer内部维度
        num_heads=4,
        num_layers=2,
        ff_dim=512,
        dropout=0.1,
        num_classes=2
    ):
        super().__init__()

        # 各模态分别投影到统一维度
        self.meta_proj = nn.Linear(input_dim, hidden_dim)
        self.text_proj = nn.Linear(input_dim, hidden_dim)
        self.graph_proj = nn.Linear(input_dim, hidden_dim)
        self.fusion_proj = nn.Linear(input_dim, hidden_dim)

        # 可学习CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # 可学习模态位置/身份编码
        self.type_embedding = nn.Parameter(torch.randn(1, 5, hidden_dim))
        # 5 = [CLS, meta, text, graph, fusion]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, h_meta, h_text, h_graph, h_fusion):
        """
        h_meta, h_text, h_graph, h_fusion: [B, input_dim]
        """
        B = h_meta.size(0)

        x_meta = self.meta_proj(h_meta).unsqueeze(1)   # [B,1,H]
        x_text = self.text_proj(h_text).unsqueeze(1)
        x_graph = self.graph_proj(h_graph).unsqueeze(1)
        x_fusion = self.fusion_proj(h_fusion).unsqueeze(1)

        cls = self.cls_token.expand(B, -1, -1)         # [B,1,H]

        x = torch.cat([cls, x_meta, x_text, x_graph, x_fusion], dim=1)  # [B,5,H]
        x = x + self.type_embedding

        h = self.transformer(x)                        # [B,5,H]
        h_cls = self.norm(h[:, 0, :])                 # [B,H]

        logits = self.classifier(h_cls)               # [B,num_classes]
        return logits

class MY_MODEL_with_Transformer(nn.Module):
    def __init__(self,
                 encoder_out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=encoder_out_size,
                                     dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=encoder_out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,
                                              hidden_size=metadata_hidden_size,
                                              output_size=encoder_out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)

        self.fusion = ModalityAttentionFusion(num_modalities=3,
                                              embed_dim=encoder_out_size,
                                            ff_dim=encoder_out_size*2,
                                           dropout=fusion_dropout)

        self.score = TransformerAggregator(input_dim=encoder_out_size,
                                           hidden_dim=256,
                                           ff_dim=4*256,
                                           num_heads=4,
                                           num_layers=2,
                                           num_classes=2)

        if is_linear_init:
            self.linear_init()

    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        self.fusion.apply(init_weights)
        self.score.apply(init_weights)

    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        logits = self.score(meta_embed,text_embed,graph_embed,fusion_embed)

        return logits
    
class MY_MODEL_without_llm_param(nn.Module):
    def __init__(self,
                 llm_path,
                 llm_freeze_param,
                 out_size=128,
                 gnn_dropout=0.1,
                 text_hidden_size=256,
                 in_num_prop_size=5,
                 in_cat_prop_size=3,
                 metadata_hidden_size=16,
                 fusion_dropout=0.3,
                 device='cuda',
                 llm_size='1B',
                 fusion_type='project_concat',
                 is_linear_init=True):
        super().__init__()
        self.device = device
        self.graph_encoder = BotRGCN(num_prop_size=in_num_prop_size,
                                     cat_prop_size=in_cat_prop_size,
                                     embedding_dimension=out_size,
                                     dropout=gnn_dropout)
        self.text_encoder = FeatureFusion2(dim1=768,
                                            dim2=768,
                                            dim3=768,
                                            hidden_size=text_hidden_size,
                                            output_size=out_size,
                                            fusion_type=fusion_type, 
                                            activation='gelu', 
                                            dropout=0.1)
        self.metadata_encoder = FeatureFusion(dim1=in_num_prop_size,
                                              dim2=in_cat_prop_size,hidden_size=metadata_hidden_size,output_size=out_size,
                                              fusion_type=fusion_type, 
                                              activation='gelu',
                                              dropout=0.1)
        

        # self.llm_model,self.pretoken_model_train,self.word_embeddings,self.vocab_size = self.init_llm_model(llm_path,llm_freeze_param)

        self.llm_model,self.pretoken_model_train = self.init_llm_model(llm_path,llm_freeze_param)

        # self.word_embeddings = self.llm_model.get_input_embeddings().weight # [32001,4096] # 32002
        # self.vocab_size = self.word_embeddings.shape[0] # 32002

        # self.mapping_m = nn.Linear(self.vocab_size + 1, 1)# 将词汇表从n+1维映射到1维 图2(c)步骤
        # self.mapping_t = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_g = nn.Linear(self.vocab_size + 1, 1)
        # self.mapping_f = nn.Linear(self.vocab_size + 1, 1)

        self.fusion = ModalityAttentionFusion(num_modalities=3,
                                            embed_dim=out_size,
                                            ff_dim=out_size*2,
                                           dropout=fusion_dropout)
        if llm_size == '1B':
            self.align_m = nn.Linear(out_size,2048)
            self.align_t = nn.Linear(out_size,2048)
            self.align_g = nn.Linear(out_size,2048)
            self.align_f = nn.Linear(out_size,2048)
        elif llm_size == '3B':
            self.align_m = nn.Linear(out_size,3072)
            self.align_t = nn.Linear(out_size,3072)
            self.align_g = nn.Linear(out_size,3072)
            self.align_f = nn.Linear(out_size,3072)
        elif llm_size == '8B':
            self.align_m = nn.Linear(out_size,4096)
            self.align_t = nn.Linear(out_size,4096)
            self.align_g = nn.Linear(out_size,4096)
            self.align_f = nn.Linear(out_size,4096)
        if is_linear_init:
            self.linear_init()
        
    def init_llm_model(self,llm_path,freeze_param):
        # LLM 模型初始化
        tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, padding_side='right')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'sep_token': '[SEP]'})

        new_tokens = ["<metadata>","<text>","<graph>","<fusion>"]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        # llm_model = AutoModelForSequenceClassification.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
        config = AutoConfig.from_pretrained(llm_path,num_labels=2)
        llm_model = AutoModelForSequenceClassification.from_config(config,  torch_dtype=torch.bfloat16)

        llm_model.resize_token_embeddings(len(tokenizer))
        llm_model.config.pad_token_id = tokenizer.pad_token_id

        pretoken_model_train=PreToken_Train(tokenizer)

        if freeze_param:
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
        for param in llm_model.score.parameters():
            param.requires_grad = True

        for name, param in llm_model.named_parameters():
            print(name, param.requires_grad)
        
        return llm_model,pretoken_model_train
    
    def linear_init(self):
        self.graph_encoder.apply(init_weights)
        self.text_encoder.apply(init_weights)
        self.metadata_encoder.apply(init_weights)
        # init_weights(self.mapping_m)
        # init_weights(self.mapping_t)
        # init_weights(self.mapping_g)
        # init_weights(self.mapping_f)
        init_weights(self.align_m)
        init_weights(self.align_t)
        init_weights(self.align_g)
        init_weights(self.align_f)
    
    def forward(self,pre_x,des,tweet,num_prop,cat_prop,edge_index,edge_type,prompt,idx):
        meta_embed = self.metadata_encoder(num_prop,cat_prop)[idx]
        text_embed = self.text_encoder(des,tweet,pre_x)[idx]
        graph_embed = self.graph_encoder(des,tweet,num_prop,cat_prop,edge_index,edge_type)[idx]
        fusion_embed = self.fusion([meta_embed,text_embed,graph_embed])

        meta_embed_aligned = self.align_m(meta_embed)
        text_embed_aligned = self.align_t(text_embed)
        graph_embed_aligned = self.align_g(graph_embed)
        fusion_embed_aligned = self.align_f(fusion_embed)

        instruction_tokens = self.pretoken_model_train(prompt,64)

        # 确保所有张量在同一设备上
        instruction_tokens = instruction_tokens.to(self.device)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            
        with torch.amp.autocast("cuda",dtype=torch.float16):

            # graph_inputs_llm = fusion_embed # [batch_size, 4096]
            meta_inputs_llm = meta_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            text_inputs_llm = text_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            graph_inputs_llm = graph_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]
            fusion_inputs_llm = fusion_embed_aligned.unsqueeze(1)# [batch_size, 1, 4096]

            # llama_word_embeddings = self.word_embeddings.unsqueeze(0) # [1,32001,4096]
            # llama_word_embeddings = llama_word_embeddings.repeat(
            # graph_inputs_llm.size(0), 1, 1).to(self.device)# 沿第一个维度复制 llama_word_embeddings,维度变为[batch_size,32001,4096]

            # cat_embedding_m = self.get_cat_embedding(meta_inputs_llm,llama_word_embeddings,"m")
            # cat_embedding_t = self.get_cat_embedding(text_inputs_llm,llama_word_embeddings,"t")
            # cat_embedding_g = self.get_cat_embedding(graph_inputs_llm,llama_word_embeddings,"g")
            # cat_embedding_f = self.get_cat_embedding(fusion_inputs_llm,llama_word_embeddings,"f")

            inputs_embeds = self.concat_instruction_modality_embeds(instruction_embeds,meta_inputs_llm,text_inputs_llm,graph_inputs_llm,fusion_inputs_llm)

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=instruction_tokens.attention_mask,
                    return_dict=True,
                )
            logits = outputs.logits

        return logits

    def get_cat_embedding(self,modality_embeddings,llama_word_embeddings,modality):
        cat_embedding = torch.cat([modality_embeddings, llama_word_embeddings], dim=1)# [batch_size,32002,4096]
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()# [batch_size,4096,32002] # .contiguous()：确保张量在内存中的存储是连续的，避免可能的数据存储问题
        if modality=='m':
            cat_embedding = self.mapping_m(cat_embedding)# [batch_size,4096,1] # self.mapping是将32002->1
        elif modality=='t':
            cat_embedding = self.mapping_t(cat_embedding)
        elif modality=='g':
            cat_embedding = self.mapping_g(cat_embedding)
        elif modality=='f':
            cat_embedding = self.mapping_f(cat_embedding)
        cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

        return cat_embedding

    def concat_instruction_modality_embeds(self,instruction_embeds,embedding_m,embedding_t,embedding_g,embedding_f):
        # <metadata>:33
        # <text>:38
        # <graph>:43
        # <fusion>:49
        front_embeds = instruction_embeds[:,:33,:]
        # embedding_m
        mid_embeds1 = instruction_embeds[:,34:38,:]
        # embedding_t
        mid_embeds2 = instruction_embeds[:,39:43,:]
        # embedding_g
        mid_embeds3 = instruction_embeds[:,44:49,:]
        # embedding_f
        back_embeds = instruction_embeds[:,50:,:]
        new_embeds = torch.cat([front_embeds,
                                embedding_m,
                                mid_embeds1,
                                embedding_t,
                                mid_embeds2,
                                embedding_g,
                                mid_embeds3,
                                embedding_f,
                                back_embeds], dim=1)

        return new_embeds