import tqdm
import torch
import numpy as np

from torch.optim import Adam
from metrics import recall_at_k, ndcg_k

class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args, logger):
        super(Trainer, self).__init__()

        self.args = args
        self.logger = logger
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        self.logger.info(f"Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}")

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch):
        self.args.train_matrix = self.args.valid_rating_matrix
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        """在测试集上评估模型性能
        
        Args:
            epoch (int): 当前轮次
            
        流程:
            1. 设置使用测试集的评分矩阵
            2. 调用iteration方法进行测试
            3. 返回测试指标(HR@k, NDCG@k等)
            
        Returns:
            tuple: (评估指标列表, 结果信息字符串)
                - 评估指标列表包含[HR@5, NDCG@5, HR@10, NDCG@10, HR@20, NDCG@20]
                - 结果信息字符串包含格式化的评估结果
        """
        # 设置当前使用的评分矩阵为测试集的评分矩阵
        self.args.train_matrix = self.args.test_rating_matrix
        # 以非训练模式调用iteration方法进行测试评估
        return self.iteration(epoch, self.test_dataloader, train=False)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        self.logger.info(original_state_dict.keys())
        new_dict = torch.load(file_name)
        self.logger.info(new_dict.keys())
        for key in new_dict:
            original_state_dict[key]=new_dict[key]
        self.model.load_state_dict(original_state_dict)

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        # import pdb; pdb.set_trace()
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HR@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HR@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        self.logger.info(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def iteration(self, epoch, dataloader, train=True):
        """每个epoch的训练或测试迭代
        Args:
            epoch: 当前轮次
            dataloader: 数据加载器
            train: 是否为训练模式，默认为True
        """
        # 根据模式设置显示的字符串
        str_code = "train" if train else "test"
        # 设置进度条
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                desc="Mode_%s:%d" % (str_code, epoch),
                                total=len(dataloader),
                                bar_format="{l_bar}{r_bar}")
        
        if train:  # 训练模式
            self.model.train()  # 设置模型为训练模式
            rec_loss = 0.0  # 初始化总损失

            for i, batch in rec_data_iter:
                # 将批次数据转移到指定设备(GPU或CPU)
                batch = tuple(t.to(self.device) for t in batch)

                # 解包批次数据
                user_ids, input_ids, answers, neg_answer, same_target = batch
                # 计算当前批次的损失
                loss = self.model.calculate_loss(input_ids, answers, neg_answer, same_target, user_ids)
                    
                # 梯度清零
                self.optim.zero_grad()
                # 反向传播
                loss.backward()
                # 参数更新
                self.optim.step()
                # 累加损失
                rec_loss += loss.item()

            # 记录训练信息
            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
            }

            # 按设定频率打印日志
            if (epoch + 1) % self.args.log_freq == 0:
                self.logger.info(str(post_fix))

        else:  # 测试/评估模式
            self.model.eval()  # 设置模型为评估模式
            pred_list = None   # 存储预测列表
            answer_list = None # 存储真实答案列表

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, answers, _, _ = batch

                # 获取模型预测输出
                recommend_output = self.model.predict(input_ids, user_ids)
                recommend_output = recommend_output[:, -1, :]  # 只取序列最后一个时间步的预测结果
                
                # 获取完整的评分预测
                rating_pred = self.predict_full(recommend_output)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                
                try:
                    # 将训练集中已存在的交互置0，避免推荐已交互过的物品
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                except: # bert4rec的特殊处理
                    rating_pred = rating_pred[:, :-1]
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                # 使用快速选择算法获取top-k推荐列表
                # 首先用argpartition获取前20个最大值的索引（时间复杂度O(n)）
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                # 获取这些索引对应的评分值
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                # 对这20个评分进行降序排序
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                # 获取最终排序后的推荐物品索引
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                # 收集所有批次的预测结果和真实答案
                if i == 0:  # 第一个批次
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:  # 后续批次，拼接结果
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

            # 计算评估指标并返回
            return self.get_full_sort_score(epoch, answer_list, pred_list)
