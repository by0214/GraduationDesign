import pandas as pd
import jieba
from gensim.models import Word2Vec
import torch
import time
import numpy as np

aspects = [ 'location_traffic_convenience',
            'location_distance_from_business_district', 
            'location_easy_to_find',
            'service_wait_time', 
            'service_waiters_attitude',
            'service_parking_convenience', 
            'service_serving_speed', 
            'price_level',
            'price_cost_effective', 
            'price_discount', 
            'environment_decoration',
            'environment_noise', 
            'environment_space', 
            'environment_cleaness',
            'dish_portion', 
            'dish_taste', 
            'dish_look', 
            'dish_recommendation']
category = {
    '位置': ['location_traffic_convenience','location_distance_from_business_district', 'location_easy_to_find'],
    '服务': ['service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed'],
    '价格': ['price_level', 'price_cost_effective', 'price_discount'],
    '环境': ['environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness'],    
    '菜品': ['dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation']
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
data_path = './data/train.csv'
test_path = './data/test.csv'
validation_path = './data/sentiment_analysis_validationset.csv'
stopwords_path = './data/stopwords.txt'
POSITIVE = 3
NEUTRAL = 2
NEGATIVE = 1

# 加载一个csv文件
def load_csv(file_path):
    return pd.read_csv(file_path, nrows=5)

# 加载停顿词
def load_stopwords(path):
    stopwords = [line.strip() for line in open(path, 'r', encoding='utf-8').readlines()]
    return stopwords

# 分词、删除停顿词
def data_clean(data, stopwords):
    print('data cleaning...')
    # data['content_cut'] = data['content'].apply(lambda x: (' '.join(jieba.cut(x))).split())
    data['content_cut'] = data['content'].apply(lambda x: jieba.lcut(x))
    data['review'] = data['content_cut'].apply(lambda x: [word for word in x if word not in stopwords])

# 提取文字与情感方面及其分值
def filter_data(df):
    print('filtering...')
    dict_lis = []
    for i, row in df.iterrows():
        data = {}
        data['cut_words'] = row['review']
        data['aspects'] = []
        for aspect in aspects:
            if row[aspect] != -2:
                data['aspects'].append((aspect, row[aspect]))
        dict_lis.append(data)
    return dict_lis

# 输入category，建立一个倒排索引
def build_index(category):
    index = {}
    for key in category:
        for value in category[key]:
            index[value] = key
    return index

inverted_index = build_index(category)

# 按照倒排索引修改字典中aspects列表中的元组的第一个元素
def modify_index(dict_lis):
    for data in dict_lis:
        for i, aspect in enumerate(data['aspects']):
            data['aspects'][i] = (inverted_index[aspect[0]], aspect[1])
    return dict_lis

# 遍历dict_lis并且把['aspects']键值列表中第0个位置相同的元组合并成一个元组，然后将它们第1个位置的数值相加
def merge_aspects(dict_lis):
    print('merging ...')
    for data in dict_lis:
        aspects_dict = {}
        for aspect in data['aspects']:
            if aspect[0] in aspects_dict:
                aspects_dict[aspect[0]] += aspect[1]
            else:
                aspects_dict[aspect[0]] = aspect[1]
        # 遍历aspects_dict, 把value>0的改为POSITIVE, value<0的改为NEGATIVE, value=0的改为NEUTRAL
        for key in aspects_dict:
            if aspects_dict[key] > 0:
                aspects_dict[key] = POSITIVE
            elif aspects_dict[key] < 0:
                aspects_dict[key] = NEGATIVE
            else:
                aspects_dict[key] = NEUTRAL
        data['aspects'] = list(aspects_dict.items())
    return dict_lis

# 新建一个Dataframe，有三个列，分别叫‘cut_words’、'aspect'、'sentiment_score'，然后将dict_lis拆分进去
def to_dataframe(dict_lis):
    print('saving...')
    df = pd.DataFrame(columns=['cut_words', 'aspect', 'sentiment_score'])
    for data in dict_lis:
        for aspect in data['aspects']:
            df.loc[len(df)] = {'cut_words': data['cut_words'], 'aspect': aspect[0], 'sentiment_score': aspect[1]}
    return df

# 文本序列化
def text_to_indices(text, vocab, max_len=128):
    indices = []
    for word in text:
        if word in vocab:
            indices.append(vocab[word])
        else:
            indices.append(vocab['<unk>'])
    if len(indices) < max_len:
        indices += [vocab['<pad>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor(indices, dtype=torch.long).to(device)

def modify_vocab(vocab):
    embedding_dim = vocab.vector_size
    weights = vocab.vectors
    pad_vector = np.zeros(embedding_dim)  
    unk_vector = np.mean(weights, axis=0)   
    if '<pad>' not in vocab:
        vocab.add_vector('<pad>', pad_vector)

    if '<unk>' not in vocab:
        vocab.add_vector('<unk>', unk_vector)
    return vocab

# 写一个函数可以将输入的文本信息转换成神经网络对应的格式,用于预测
def text_to_tensor(text, aspect, model):
    # 首先对评论进行分词和数据清洗
    text_lis = jieba.lcut(text)
    stopwords = load_stopwords(stopwords_path)
    text_lis = [word for word in text_lis if word not in stopwords]

    # 然后就是进行序列化
    # model = Word2Vec.load('./Word2Vec/model_myself_300.model')
    # vocab = modify_vocab(model.wv).key_to_index
    vocab = model.wv.key_to_index
    text_indices = text_to_indices(text_lis, vocab)
    aspect_indices = text_to_indices([aspect], vocab)
    
    return text_indices.unsqueeze(0), aspect_indices.unsqueeze(0)


if __name__ == '__main__':
    time_start = time.time()
    data = load_csv(data_path)
    stopwords = load_stopwords(stopwords_path)
    data_clean(data, stopwords)
    dict_lis = filter_data(data)
    dict_lis = modify_index(dict_lis)
    dict_lis = merge_aspects(dict_lis)
    df = to_dataframe(dict_lis)
    df.to_csv('./data/cut_trainset.csv', index=False)
    time_end = time.time()
    print('preprocess finished!')
    print('time cost:', time_end - time_start, 's')

    # 训练一个300维的模型
    # data = pd.read_csv(data_path)
    # stopwords = load_stopwords(stopwords_path)
    # data_clean(data, stopwords)
    # sentences = data['review']
    # model = Word2Vec(sentences, vector_size=300)
    # model.train(sentences, total_examples=len(sentences), epochs=10)
    # model.save('./Word2Vec/model_myself_300.model')
    
    # 测试text_to_tensor函数
    # text = "我觉得这家餐厅交通很便利。"
    # aspect = "位置"
    # text_tensor, aspect_tensor  = text_to_tensor(text, aspect)

    # time_start = time.time()
    # data = load_csv(validation_path)
    # stopwords = load_stopwords(stopwords_path)
    # data_clean(data, stopwords)
    # dict_lis = filter_data(data)
    # dict_lis = modify_index(dict_lis)
    # dict_lis = merge_aspects(dict_lis)
    # df = to_dataframe(dict_lis)
    # df.to_csv('./data/cut_valiset.csv', index=False)
    # time_end = time.time()
    # print('preprocess finished!')
    # print('time cost:', time_end - time_start, 's')

    # df = load_csv('./data/sentiment_analysis_trainingset.csv')
    # df.to_csv('data_sample')

    

