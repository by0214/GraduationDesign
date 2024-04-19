'''
这个脚本呢主要是用来实现如下功能：
1. 读入数据
2. 把这批数据的权重向量求出来
3. 对于权重向量进行分割,分割为25个子集
    最在意环境的人、最在意位置的人...
    第二在意环境的人、第二在意位置的人...
    ...
    最不在意环境的人、最不在意位置的人...
4. 对于每一个集合分别进行html + DT + Bayesian, 看求出的向量是否满足集合的条件计算正确率
'''
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import optimism as op
from sklearn.tree import DecisionTreeClassifier
import time
from HMPM.hmpm import HMPM_MODEL
from tqdm import tqdm
import logging
from logging.handlers import RotatingFileHandler

# 配置日志格式，包括时间戳、日志级别和日志消息
log_format = '%(asctime)s - %(levelname)s - %(message)s'
handler = RotatingFileHandler('./result/training_log.txt', maxBytes=128*1024, backupCount=5)
handler.setFormatter(logging.Formatter(log_format))
logger = logging.getLogger('training')
logger.setLevel(logging.INFO)  # 设置日志记录级别
logger.addHandler(handler)  # 给logger添加handler
logger.info('=' * 100)

# 一些超参数
TRAIN_NUM = 5000      
SKIP_ROW_NUM = 4410000
logger.info(f'TRAIN_NUM:{TRAIN_NUM}')
logger.info(f'SKIP_ROW_NUM:{SKIP_ROW_NUM}')              
aspect_lis = ['environment','location','dish','service','price','rating']
csv_file = './testData/ratings.csv'
logger.info('开始导入模型')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save_path = './newmodel/trained_model/model_bert_wwm.pth'
tokenizer_save_path = './newmodel/tokenizer1e-wwm'
tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)
model = BertForSequenceClassification.from_pretrained('./newmodel/chinese_wwm_ext_pytorch', num_labels=3)
model.load_state_dict(torch.load(model_save_path, map_location=device))
logger.info(f"模型{model_save_path}加载完毕")
model.to(device)
logger.info(f'使用{device}进行运算')

def load_data(csv_path, train_num, skip_num):
    '''
    从csv文件的skip_num处加载train_num条数据
    返回值：
        文本列表，分数列表
    
    '''
    logger.info("=" * 50)
    logger.info('开始读取数据')
    df = pd.read_csv(csv_path, skiprows=skip_num, names=['userId', 'restId', 'rating', 'rating_env', 'rating_flavor', 'rating_service', 'timestamp', 'comment'])
    df = df[:train_num]
    df = df.dropna(subset=['comment', 'rating'])

    texts_ = df['comment'].tolist()
    star_lis = df['rating'].tolist()

    # 清洗文本
    texts = []
    for text in tqdm(texts_, desc="文本清洗", total=len(texts)):
        text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefA-Za-z0-9]', '', text)
        texts.append(text)

    # 映射分数
    star_lis = [0.6 * i for i in tqdm(star_lis, desc="映射分数", total=len(star_lis))]
    logger.info('数据读取完毕')
    logger.info("=" * 50)
    return texts, star_lis

def caculate_weight_matrix(texts, star_lis):
    '''
    从文本列表和分数列表计算出分数矩阵和权重矩阵
    返回值：
        score_matrix(最后一列是综合评分)
        final_weight_matrix
    
    '''
    logger.info("=" * 50)
    logger.info("开始计算分数矩阵以及权重矩阵")
    global model, tokenizer, device
    optimizer = op.weight_optimizer(model, tokenizer, texts, star_lis, device)
    score_matrix = optimizer.score_matrix
    final_weight_matrix = optimizer.optimize()
    logger.info("分数矩阵")
    logger.info(score_matrix)
    logger.info("权重矩阵")
    logger.info(final_weight_matrix)
    logger.info("=" * 50)
    return score_matrix, final_weight_matrix

def split_user_to_25(final_weight_matrix):
    '''
    把这个权重矩阵分为25个部分, 每个部分代表一类用户群体
    返回值：
        一个(5 * 5)长度的列表,矩阵下标(x, y)对应列表的y(第几个方面)+index_matrix[i, j] * 5(这方面排名第几)位置的集合, 集合存储x(哪个用户)
        列表位置的含义：[[环境权重第一集合], [位置权重第一集合], ... [口味权重第二集合] ... [价格权重第五集合]]
    '''
    # 结果列表
    logger.info("开始用户分类")
    logger.info("=" * 50)
    result_set = [[] for _ in range(25)]

    # 首先要构造一个排序矩阵, 矩阵第(i, j)位置的值为，原矩阵(i, j)排序后的下标
    index_matrix = np.zeros(final_weight_matrix.shape)
    for i in tqdm(range(final_weight_matrix.shape[0]), desc='用户分类'):
        indices = np.argsort(-final_weight_matrix[i])
        index_matrix[i] = np.argsort(indices)
    logger.info('下标矩阵')
    logger.info(index_matrix)

    for user_id in range(final_weight_matrix.shape[0]):
        for aspect_id in range(final_weight_matrix.shape[1]):
            result_set[int(aspect_id + index_matrix[user_id][aspect_id] * 5)].append(user_id)
    logger.info("用户分类情况")
    logger.info(result_set)
    return result_set

def train(user_set, weight_matrix, score_matrix):
    '''
    通过user_set中存储的分类信息, 构建25个矩阵, 分别给DT和html训练, 把结果存起来, 并且把25个分数矩阵存储起来以便日后给Bayesian进行训练
    '''
    global aspect_lis, model, tokenizer, device, TRAIN_NUM
    hmtm_res = []
    dt_res = []
    logger.info(f'=' * 50)
    logger.info(f'开始训练')
    for i, users in enumerate(user_set):
        if len(users) == 0:
            continue
        # 构建相关矩阵
        cur_weight_matrix = [weight_matrix[user_id] for user_id in users]
        cur_score_matrix = [score_matrix[user_id] for user_id in users]
        df_scores = pd.DataFrame(cur_score_matrix, columns=aspect_lis)
        df_scores.to_csv(f'./result/user_scores_{TRAIN_NUM}_{i}.csv', index=False)
        cur_weight_matrix = np.array(cur_weight_matrix)
        cur_score_matrix = np.array(cur_score_matrix)

        # 训练hmpm
        cur_hmpm_model = HMPM_MODEL(cur_weight_matrix)
        hmtm_res.append(np.round(cur_hmpm_model.optimize(), 7))

        # 训练DT
        clf = DecisionTreeClassifier()
        X = cur_score_matrix[:, :-1]
        y = cur_score_matrix[:, -1]
        y = pd.Categorical(y).codes
        clf.fit(X, y)
        feature_importances = clf.feature_importances_
        dt_res.append(np.round(np.array(feature_importances), 7))
    logger.info('html结果')
    logger.info(hmtm_res)
    logger.info('DT结果')
    logger.info(dt_res)
    logger.info('=' * 50)
    df_hmpm = pd.DataFrame(np.array(hmtm_res), columns=aspect_lis[:-1])
    df_dt = pd.DataFrame(np.array(dt_res), columns=aspect_lis[:-1])
    df_hmpm.to_csv(f'./result/hmpm_res_{TRAIN_NUM}.csv', index=False)
    df_dt.to_csv(f'./result/dt_res_{TRAIN_NUM}.csv', index=False)

def calculate_acc(csv_path):
    '''
    读取csv文件检查准确率
    '''
    df = pd.read_csv(csv_path)
    df_data = df.values.tolist()
    list_tosave = []
    for i, row in enumerate(df_data):
        row = np.array(row)
        np.round(row, decimals=7)
        row_sort_index = np.argsort(-row)
        list_tosave.append(row_sort_index)
    df_tosave = pd.DataFrame(list_tosave, columns=aspect_lis[:-1])
    name = csv_path.split('/')[-1].split('.')[0]
    df_tosave.to_csv(f'./result/{name}_sort.csv', index=False)


if __name__ == '__main__':
    texts, star_lis = load_data(csv_file, TRAIN_NUM, SKIP_ROW_NUM)
    score_matrix, final_weight_matrix = caculate_weight_matrix(texts, star_lis)
    user_set = split_user_to_25(final_weight_matrix)
    train(user_set, final_weight_matrix, score_matrix)

    csv_file = [f'./result/hmpm_res_{TRAIN_NUM}.csv', f'./result/dt_res_{TRAIN_NUM}.csv']
    for csv in csv_file:
        calculate_acc(csv)