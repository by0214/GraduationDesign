import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

class SentimentDataset(Dataset):
    def __init__(self, csv_path, vocab, max_len):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.vocab = vocab.key_to_index
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['cut_words']
        aspect = self.df.iloc[idx]['aspect']
        sentiment_score = self.df.iloc[idx]['sentiment_score']

        text_indices = self.text_to_indices(text, self.vocab, self.max_len)
        aspect_indices = self.text_to_indices([aspect], self.vocab, self.max_len)

        return (text_indices, aspect_indices), sentiment_score - 1
    
    @staticmethod
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

# def modify_vocab(vocab):
#     embedding_dim = vocab.vector_size
#     weights = vocab.vectors
#     pad_vector = np.zeros(embedding_dim)  
#     unk_vector = np.mean(weights, axis=0)   
#     if '<pad>' not in vocab:
#         vocab.add_vector('<pad>', pad_vector)

#     if '<unk>' not in vocab:
#         vocab.add_vector('<unk>', unk_vector)
#     return vocab

# if __name__ == '__main__':
#     from gensim.models import Word2Vec
#     from torch.utils.data import DataLoader
#     csv_path = './data/cut_trainset.csv'
#     model = Word2Vec.load('./Word2Vec/model_myself.model')
#     vocab = modify_vocab(model.wv)
#     maxlen = 128
    
#     dataset = SentimentDataset(csv_path, vocab, maxlen)
#     data_loader = DataLoader(dataset, batch_size=3, shuffle=False)
    
#     for i, (inputs, labels) in enumerate(data_loader):
#         print(inputs[1])
#         print(labels)
#         print('---------------------------------------------')
#         if i > 1:
#             break
    
