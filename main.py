import os

import torch
import numpy as np

from model import MODEL_DICT
from trainers import Trainer
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from dataset import get_seq_dic, get_dataloder, get_rating_matrix

def main():
    # 解析命令行参数
    args = parse_args()
    # 构建日志文件路径
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    # 初始化日志记录器
    logger = set_logger(log_path)

    # 设置随机种子，确保实验可复现
    set_seed(args.seed)
    # 检查并创建输出目录
    check_path(args.output_dir)

    # 设置可见的GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # 判断是否使用CUDA
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # 获取用户序列数据及相关统计信息
    seq_dic, max_item, num_users = get_seq_dic(args)
    # 设置物品和用户的数量(加1是为了包含padding的0)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    # 设置模型检查点和相似目标路径
    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name+'_same_target.npy')
    # 获取数据加载器
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)

    # 记录参数配置和模型信息
    logger.info(str(args))
    # 根据指定的模型类型初始化模型
    model = MODEL_DICT[args.model_type.lower()](args=args)
    logger.info(model)
    # 初始化训练器
    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)

    # 获取验证集和测试集的评分矩阵
    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    if args.do_eval:  # 仅进行评估模式
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            # 加载预训练模型进行测试
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            logger.info(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0)

    else:  # 训练模式
        # 初始化早停机制
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        # 开始训练循环
        for epoch in range(args.epochs):
            # 训练一个epoch
            trainer.train(epoch)
            # 在验证集上评估
            scores, _ = trainer.valid(epoch)
            # 使用MRR指标进行早停判断
            early_stopping(np.array([scores[-1]]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        # 训练结束后的测试评估
        logger.info("---------------Test Score---------------")
        # 加载最佳模型进行测试
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))#从文件中读取之前保存的最佳模型参数
        scores, result_info = trainer.test(0)

    # 记录实验名称和最终结果
    logger.info(args.train_name)
    logger.info(result_info)
    
if __name__ == "__main__":
    main()
