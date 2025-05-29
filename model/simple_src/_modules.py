import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
#from torch_dct import dct,idct
import pytorch_wavelets as ptwt
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()

        hidden_size = args.hidden_size
        inner_size = 4 * args.hidden_size

        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(args.hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states +input_tensor)#+ input_tensor[:,:,:64])

        return hidden_states

class HighFrequencySignalScale(nn.Module):
    """高频信号缩放模块
    
    用于调整小波变换后高频分量的重要性。通过可学习的参数对不同尺度的高频信号进行加权。
    
    Args:
        args: 配置参数对象，包含以下字段:
            - hidden_size: 隐藏层维度
            - wave: 小波基函数类型(如'haar')
            - max_seq_length: 最大序列长度
            - decomp_level: 小波分解层数
    """
    def __init__(self,args):
        super(HighFrequencySignalScale, self).__init__()
        # 存储每层分解的可学习权重参数
        self.ww = nn.ParameterList()
        self.ww1 = nn.ParameterList() 
        # 全局缩放因子，形状为[hidden_size]
        self.srt = nn.Parameter(torch.rand(args.hidden_size))
        
        # 获取小波基函数
        wave = pywt.Wavelet(args.wave)
        # 获取滤波器长度
        filter_len = wave.dec_len
        # 初始化序列长度
        init_len = args.max_seq_length
        
        # 为每一层分解创建可学习的权重矩阵
        for i in range(args.decomp_level):
            # 计算每一层的序列长度
            init_len = (init_len + filter_len-1)//2
            # 添加形状为[init_len, hidden_size]的可学习参数
            self.ww.append(nn.Parameter(torch.rand(init_len, args.hidden_size)))
            
        # 记录分解层数    
        self.max_lv = args.decomp_level
        
    def forward(self, cd, ca):
        """前向传播函数
        
        Args:
            cd: 小波分解的细节系数(高频分量)列表
            ca: 小波分解的近似系数(低频分量)
            
        Returns:
            tuple: (ca, cd)
                - ca: 调整后的近似系数
                - cd: 调整后的细节系数列表
        """
        # 对每一层的高频分量进行缩放
        for i in range(self.max_lv):
            # 1. permute改变维度顺序以进行矩阵乘法
            # 2. 应用可学习权重ww[i]和全局缩放因子srt
            # 3. permute恢复原始维度顺序
            cd[i] = torch.mul(torch.mul(cd[i].permute(0,2,1), self.ww[i]), self.srt**2).permute(0,2,1)
            
        # 注释掉的代码是对低频分量的处理
        #ca = torch.mul(torch.mul(ca,self.ww[i]).permute(0,2,1),self.ww1[i]).permute(0,2,1)
        
        return ca, cd
        
class FrequencyLayer(nn.Module):
    """频域处理层
    
    使用小波变换对输入序列进行频域分解和重构，包含高频信号的自适应调整。
    
    Args:
        args: 配置参数对象，包含以下字段:
            - attention_probs_dropout_prob: dropout概率
            - hidden_size: 隐藏层维度
            - c: 频率截断参数
            - decomp_level: 小波分解层数
            - wave: 小波基函数类型
            - n: 其他超参数
    """
    def __init__(self, args):
        super(FrequencyLayer, self).__init__()
        # Dropout层，用于防止过拟合
        self.out_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        # 层归一化，用于特征标准化
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        # 频率截断参数
        self.c = args.c // 2 + 1
        # 小波分解层数
        self.lv = args.decomp_level
        # 高频信号的可学习缩放因子
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, args.hidden_size))
        # 前向小波变换
        self.fwd = ptwt.DWT1DForward(wave=args.wave, J=args.decomp_level)
        # 逆向小波变换
        self.inv = ptwt.DWT1DInverse(wave=args.wave)
        # 其他超参数
        self.cdn = args.n
        # 高频信号缩放模块
        self.scale = HighFrequencySignalScale(args)
        # 是否仅保留低频信息
        self.not_restore = args.not_restore

    def forward(self, input_tensor):
        """前向传播函数
        
        Args:
            input_tensor: 输入张量，形状为[batch, seq_len, hidden]
            
        Returns:
            hidden_states: 处理后的特征表示
        """
        # 获取输入张量的维度信息
        batch, seq_len, hidden = input_tensor.shape
        
        # 进行小波变换，得到近似系数(ca)和细节系数(cd)
        ca, cd = self.fwd(input_tensor.permute(0,2,1))
        # 对高频信号进行自适应调整
        ca, cd = self.scale(cd, ca)
        
        # 进行逆变换重构信号
        low_pass = self.inv((ca,cd)).permute(0,2,1)
        # 计算高频部分
        high_pass = input_tensor - low_pass
        
        # 根据配置选择是否保留高频信息
        if self.not_restore:
            sequence_emb_fft = low_pass  # 只保留低频部分
        else:
            # 结合低频和加权后的高频
            sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass
        
        # Dropout和残差连接
        hidden_states = self.out_dropout(sequence_emb_fft)
        # 层归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    
class DWTRecLayer(nn.Module):
    """DWTRec的基本层结构
    
    包含频域处理层(FrequencyLayer)，用于处理序列中的频域信息。
    
    Args:
        args: 配置参数对象，包含以下字段:
            - alpha: 平衡参数，用于调节不同组件的贡献
    """
    def __init__(self, args):
        super(DWTRecLayer, self).__init__()
        self.args = args
        # 初始化频域处理层，用于序列的频域分解和重构
        self.filter_layer = FrequencyLayer(args)
        # 平衡参数，用于调节不同组件的贡献
        self.alpha = args.alpha
        
    def forward(self, input_tensor):
        """前向传播函数
        
        Args:
            input_tensor: 输入张量，形状为[batch_size, seq_len, hidden_size]
            
        Returns:
            hidden_states: 经过频域处理后的特征表示
        """
        # 通过频域处理层处理输入序列
        dsp = self.filter_layer(input_tensor)
        
        # 直接返回频域处理的结果
        hidden_states = dsp 
        
        return hidden_states
    
class DWTRecBlock(nn.Module):
    """DWTRec的基本块结构
    
    包含一个DWTRecLayer和一个前馈网络层(FeedForward)，用于串行处理输入特征。
    
    Args:
        args: 配置参数对象，用于初始化内部组件
    """
    def __init__(self, args):
        super(DWTRecBlock, self).__init__()
        # 初始化DWTRec基本层，处理频域信息
        self.layer = DWTRecLayer(args)
        # 初始化前馈网络层，用于特征转换
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states):
        """前向传播函数
        
        Args:
            hidden_states: 输入的隐藏状态，形状为[batch_size, seq_len, hidden_size]
            
        Returns:
            feedforward_output: 经过频域处理和前馈网络处理后的特征表示
        """
        # 1. 通过DWTRecLayer处理输入
        layer_output = self.layer(hidden_states)
        # 2. 通过前馈网络进一步处理
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


