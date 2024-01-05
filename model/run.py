import torch
from torch.utils.data import DataLoader
from cnn_gate_aspect import CNN_Gate_Aspect_model
from mydataset import SentimentDataset
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import torch
from train_eval import train, evaluate
import matplotlib.pyplot as plt
import random


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

def plotdict(data:dict, epoch, title):
    plt.figure()
    plt.xlabel('Epoch')
    plt.title(title)
    x = [i+1 for i in range(epoch)]
    for k, v in data.items():
        plt.plot(x, v, label=k, color=(random.random(), random.random(), random.random()))
    plt.savefig(f'plot_{title}.png', format='png')  

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


# 初始化数据集的参数
model_word  = Word2Vec.load('./Word2Vec/model_myself_300.model')
vocab = modify_vocab(model_word.wv)
# model_word = KeyedVectors.load_word2vec_format('./Word2Vec/word2vec_779845.bin', binary=True)
# vocab = modify_vocab(model_word)
weights = vocab.vectors
embedding_matrix = torch.tensor(weights, dtype=torch.float).to(device)
max_len = 128


# 初始化训练参数
config = {
    'learning_rate': 0.01,
    'batch_size': 32,
    'num_epochs': 30,
    'train_path': './data/cut_trainset.csv',
    'test_path': './data/cut_testset.csv',
    'vali_path': './data/cut_valiset.csv'
}

args = {
    'embed_num': weights.shape[0],
    'embed_dim': weights.shape[1],
    'class_num': 3,
    'aspect_num': 5,
    'kernel_num': 100,
    'kernel_sizes': [3, 4, 5],
    'embedding': embedding_matrix,
    'aspect_embed_dim': weights.shape[1],
    'aspect_embedding': embedding_matrix
}

# 加载数据集
dataset = SentimentDataset(config['train_path'], vocab, max_len)
data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

testset = SentimentDataset(config['test_path'], vocab, max_len)
test_loader = DataLoader(testset, batch_size=config['batch_size'], shuffle=True)

validationset = SentimentDataset(config['vali_path'], vocab, max_len)
validation_loader = DataLoader(validationset, batch_size=config['batch_size'], shuffle=True)

# 初始化神经网络
model = CNN_Gate_Aspect_model(args).to(device) 
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=config['learning_rate'], weight_decay=0, lr_decay=0)

loss_dict, acc_dict, epoch = train(model, data_loader, validation_loader, test_loader, config, criterion, optimizer, device)
temp = evaluate(model, test_loader, criterion, device)
print(temp)
plotdict(loss_dict, epoch, 'loss')
plotdict(acc_dict, epoch, 'acc')


