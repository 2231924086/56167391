import torch
import torch.nn as nn
from model._modules import LayerNorm
from torch.nn.init import xavier_uniform_


class SequentialRecModel(nn.Module):
    def __init__(self, args):
        super(SequentialRecModel, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.user_embeddings = nn.Embedding(args.num_users, args.user_hidden_size, padding_idx=0)
        self.batch_size = args.batch_size
        # self.concat_projection = nn.Linear(args.hidden_size + args.user_hidden_size, args.hidden_size)#用于拼接序列嵌入和用户嵌入的投影层

        # self.user_freq_mlp = nn.Sequential(
        # nn.Linear(args.user_hidden_size, args.user_hidden_size // 2),
        # nn.ReLU(),
        # nn.Linear(args.user_hidden_size // 2, args.decomp_level)
        # )

        self.user_freq_mlp = nn.Sequential(
        nn.Linear(args.user_hidden_size, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, args.decomp_level)
        )

    def add_position_embedding(self, sequence, user_ids):
        """添加位置编码到输入序列

        将item embeddings和position embeddings相加，然后进行归一化和dropout处理。

        Args:
            sequence: 输入的物品ID序列，形状为[batch_size, seq_len]

        Returns:
            sequence_emb: 添加位置编码后的序列表示，形状为[batch_size, seq_len, hidden_size]
        """
        # 获取序列长度
        seq_length = sequence.size(1)

        # 生成位置ID，从0到seq_length-1
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        # 扩展维度以匹配输入sequence的形状
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        # 获取物品的嵌入表示
        item_embeddings = self.item_embeddings(sequence)
        # 获取位置的嵌入表示
        position_embeddings = self.position_embeddings(position_ids)

        # 获取用户的嵌入表示
        user_embeddings = self.user_embeddings(user_ids)  # [batch, user_hidden_size]
        user_embeddings = user_embeddings.unsqueeze(1).expand(-1, seq_length, -1)  # [batch, seq_len, user_hidden_size]

        user_embeddings = self.user_embeddings(user_ids)  # [batch, user_hidden_size]
        user_embeddings = user_embeddings.unsqueeze(1).expand(-1, seq_length, -1)  # [batch, seq_len, user_hidden_size]

        # 将物品嵌入和位置嵌入相加
        sequence_emb = item_embeddings + position_embeddings

        # sequence_emb += user_embeddings

        # concatenated_emb = torch.cat([sequence_emb, user_embeddings], dim=-1)  # [batch, seq_len, hidden*2]
        # sequence_emb = self.concat_projection(concatenated_emb)  # [batch, seq_len, hidden]

        # 层归一化，标准化特征分布
        sequence_emb = self.LayerNorm(sequence_emb)
        # Dropout，防止过拟合
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def init_weights(self, module):
        """ Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_bi_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-head attention."""

        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64

        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""

        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(self, input_ids, all_sequence_output=False):
        pass

    def predict(self, input_ids, user_ids, all_sequence_output=False):
        return self.forward(input_ids, user_ids, all_sequence_output)

    def calculate_loss(self, input_ids, answers):
        pass

