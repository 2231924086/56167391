import copy
import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, DWTRecBlock

class DWTRecEncoder(nn.Module):
    """DWTRec编码器模块
    
    该编码器使用多个DWTRecBlock块来处理序列数据
    
    Args:
        args: 配置参数对象，包含模型的超参数设置
    """
    def __init__(self, args):
        super(DWTRecEncoder, self).__init__()
        self.args = args
        # 初始化DWTRecBlock，并创建指定数量的副本组成编码层
        block = DWTRecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=False):
        """前向传播函数
        
        Args:
            hidden_states: 输入的隐藏状态，形状为[batch_size, seq_len, hidden_size]
            output_all_encoded_layers: 是否输出所有编码层的结果，默认False
        
        Returns:
            all_encoder_layers: 包含编码器各层输出的列表。
                              如果output_all_encoded_layers为True，返回所有层的输出；
                              否则只返回最后一层的输出
        """
        # 初始化编码层列表，将输入加入作为第一个元素
        all_encoder_layers = [ hidden_states ]
        
        # 依次通过每个DWTRecBlock进行处理
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                
        # 如果不需要所有层的输出，只保存最后一层的结果
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
            
        return all_encoder_layers

class DWTRecModel(SequentialRecModel):
    """DWTRec模型类
    
    继承自SequentialRecModel的推荐系统模型，结合了频域和空域的双重注意力机制。
    
    Args:
        args: 配置参数对象，包含模型的超参数设置
    """
    def __init__(self, args):
        super(DWTRecModel, self).__init__(args)
        self.args = args
        # 初始化层归一化，用于特征标准化
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        # Dropout层用于防止过拟合
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        # 初始化DWTRec编码器
        self.item_encoder = DWTRecEncoder(args)
        # 应用权重初始化
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        """前向传播函数

        Args:
            input_ids: 输入序列的ID，形状为[batch_size, seq_len]
            user_ids: 用户ID，可选参数
            all_sequence_output: 是否输出所有层的结果，默认False

        Returns:
            sequence_output: 序列的编码表示。
                           如果all_sequence_output为True，返回所有层的输出；
                           否则只返回最后一层的输出
        """
        # 添加位置编码
        sequence_emb = self.add_position_embedding(input_ids)
        # 通过编码器处理序列
        item_encoded_layers = self.item_encoder(sequence_emb,
                                              output_all_encoded_layers=True)
        
        # 根据需求返回相应的输出
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        """计算模型损失

        Args:
            input_ids: 输入序列ID
            answers: 正样本目标ID
            neg_answers: 负样本目标ID
            same_target: 相似目标序列(用于对比学习)
            user_ids: 用户ID

        Returns:
            loss: 计算得到的交叉熵损失值
        """
        # 获取序列的编码输出
        seq_output = self.forward(input_ids)
        # 只使用序列最后一个时间步的输出进行预测
        seq_output = seq_output[:, -1, :]
        # 获取所有物品的嵌入权重
        item_emb = self.item_embeddings.weight
        # 计算预测分数
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        # 使用交叉熵计算损失
        loss = nn.CrossEntropyLoss()(logits, answers)

        return loss

