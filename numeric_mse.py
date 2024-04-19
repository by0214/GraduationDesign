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

# 写一个计算均方误差的函数
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# 计算RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

# 计算MAE
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

TRAIN_NUM = [100, 500, 1000, 2000]
SKIP_ROW_NUM = 4414000
aspect_lis = ['environment','location','dish','service','price','rating']

# 数据采集脚本
def collect_data_script(train_num, skip_num):
    '''
    一条记录的规定格式是：
    time skip_num train_num y_true y_pred_dt y_pred_hmpm y_pred_bayesian dt_res hmpm_res bayesian_res 
    '''
    headers = ['time', 'skip_num', 'train_num', 'y_true', 'y_pred_dt', 'y_pred_hmpm', 'y_pred_bayesian', 'dt_res', 'hmpm_res', 'bayesian_res']
    record = []
    t_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    record = [t_now, skip_num, train_num]

    # 读入数据
    csv_file = './testData/ratings.csv'
    colunm_names = ['userId', 'restId', 'rating', 'rating_env', 'rating_flavor', 'rating_service', 'timestamp', 'comment']
    print(f'开始读取{csv_file}')
    df = pd.read_csv(csv_file, skiprows=skip_num, names=colunm_names)
    df = df[:train_num]
    print("读取完毕")

    # 丢弃其中缺失的列
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

    # 参数配置
    print('开始配置参数')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = './newmodel/trained_model/model_bert_wwm.pth'
    tokenizer_save_path = './newmodel/tokenizer1e-wwm'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)
    model = BertForSequenceClassification.from_pretrained('./newmodel/chinese_wwm_ext_pytorch', num_labels=3)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f"模型{model_save_path}加载完毕")
    model.to(device)
    print(f'使用{device}进行运算')

    '''
    所谓的均方误差说的是权重的均值，也就是优化模型输出的均值
    如何计算各个模型的均方误差呢：
    利用各个模型的现有权重与优化模型的输出的均值来算
    '''
    # 优化解
    optimizer = op.weight_optimizer(model, tokenizer, texts, star_lis, device)
    score_matrix = optimizer.score_matrix
    final_weight_matrix = optimizer.optimize()
    y_true = np.mean(final_weight_matrix, axis=0)
    record.append(y_true)
    print(f'均值向量求解完毕,y_true:{y_true}')

    # 训练DT
    print('开始训练DT')
    X = score_matrix[:, :-1]
    y = score_matrix[:, -1]
    y = pd.Categorical(y).codes
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    feature_importances = clf.feature_importances_
    y_pred_dt = feature_importances
    record.append(y_pred_dt)
    print(f'DT训练完毕,y_pred_dt:{y_pred_dt}')

    # hmpm
    hmpm_model = HMPM_MODEL(final_weight_matrix)
    y_pred_hmpm = hmpm_model.optimize()
    record.append(y_pred_hmpm)
    print(f'hmpm求解完毕,y_pred_hmpm:{y_pred_hmpm}')

    # bayesian要手工
    y_pred_bayesian = [0, 0, 0, 0, 0]
    record.append(y_pred_bayesian)

    # dt_res
    record.append([mse(y_true, y_pred_dt), rmse(y_true, y_pred_dt), mae(y_true, y_pred_dt)])

    # hmpm_res
    record.append([mse(y_true, y_pred_hmpm), rmse(y_true, y_pred_hmpm), mae(y_true, y_pred_hmpm)])

    # beysian_res
    record.append([0, 0, 0])

    df_ = pd.DataFrame([record], columns=headers)
    print("=" * 100)
    print(df_)
    print("=" * 100)
    df_.to_csv('./result/records_mse.csv', mode='a', index=False, header=False)

if __name__ == '__main__':
    for i in range(10):
        for t_num in TRAIN_NUM:
            collect_data_script(t_num, SKIP_ROW_NUM)
    SKIP_ROW_NUM -= 2000