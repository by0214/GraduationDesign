import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')
import transformers
transformers.logging.set_verbosity_error()


# 定义数据集
class SentimentDataset(Dataset):
    def __init__(self, tokenizer, texts, aspects, labels, max_length=128):
        self.texts = texts
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        aspect = self.aspects[idx]
        label = self.labels[idx]
        # 将文本和方面合并
        encoding = self.tokenizer.encode_plus(
            text, 
            aspect, 
            add_special_tokens=True, 
            max_length=self.max_length, 
            return_token_type_ids=False, 
            padding='max_length', 
            return_attention_mask=True, 
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten().to(device),
            'attention_mask': encoding['attention_mask'].flatten().to(device),
            'labels': torch.tensor(label, dtype=torch.long).to(device)
        }
    
# 初始化分词器
tokenizer = BertTokenizer.from_pretrained('./newmodel/chinese_wwm_ext_pytorch')

# 加载预训练的 BERT 模型
model = BertForSequenceClassification.from_pretrained('./newmodel/chinese_wwm_ext_pytorch', num_labels=3)
model.to(device)

# 加载数据
train_data_path = './bert_data/bert_trainset.csv'
vali_data_path = './bert_data/bert_valiset.csv'
df_train = pd.read_csv(train_data_path)
df_vali = pd.read_csv(vali_data_path)
print('加载数据完成')

 
# 准备数据集
train_texts = df_train['content'].str.strip('"').values.tolist()
train_aspects = df_train['aspect'].tolist()
train_labels = [i - 1 for i in df_train['sentiment_score'].tolist()]
val_texts = df_vali['content'].tolist()
val_aspects = df_vali['aspect'].tolist()
val_labels = [i - 1 for i in df_vali['sentiment_score'].tolist()]
print('开始准备数据集...')

train_dataset = SentimentDataset(tokenizer, train_texts, train_aspects, train_labels)
val_dataset = SentimentDataset(tokenizer, val_texts, val_aspects, val_labels)

# 训练参数
batch_size = 32
learning_rate = 3e-5
num_epochs = 3

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 优化器
# optimizer = AdamW(model.parameters(), lr=learning_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print('开始训练')

# 训练模型
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# 保存
model_save_path = "model_bert_wwm.pth"
torch.save(model.state_dict(), model_save_path)

tokenizer.save_pretrained("tokenizerwwm")
