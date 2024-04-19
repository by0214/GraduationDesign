from scipy.optimize import linprog, minimize, LinearConstraint
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
transformers.logging.set_verbosity_error()


class weight_optimizer():
    def __init__(self, model, tokenizer, texts, star_lis, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.texts = texts
        self.star_lis = star_lis
        self.score_matrix = self.get_sentiment_score()
        self.final_weight_matrix = np.zeros((len(texts), 5))

    '''
    得到分数矩阵
    '''
    def get_sentiment_score(self, aspect_lis =['环境', '位置', '菜品', '服务', '价格']):
        sentiment_score_matrix = np.zeros((len(self.texts), len(aspect_lis) + 1))
        for text in tqdm(self.texts, total=len(self.texts), desc="分析文本情感"):
            for i in range(len(aspect_lis)):
                encoding = self.tokenizer.encode_plus(
                        text, 
                        aspect_lis[i], 
                        add_special_tokens=True, 
                        max_length=128, 
                        return_token_type_ids=False, 
                        padding='max_length', 
                        return_attention_mask=True, 
                        return_tensors='pt',
                        truncation=True
                )

                test_dict = {
                            'input_ids': encoding['input_ids'].flatten().unsqueeze(0).to(self.device),
                            'attention_mask': encoding['attention_mask'].flatten().unsqueeze(0).to(self.device)
                        }

                outputs = self.model(test_dict['input_ids'], attention_mask=test_dict['attention_mask'])
                result = torch.argmax(outputs.logits, dim=1)
                sentiment_score_matrix[self.texts.index(text), i] = result.item() + 1
            sentiment_score_matrix[self.texts.index(text), -1] = self.star_lis[self.texts.index(text)]
        return sentiment_score_matrix

    '''
    针对一条评论
    s: [score1, score2, ... score5, totalscore]
    k: regularization parameter
    w: [weight1, weight2, ... weight5]
    '''
    def create_objective(s, k=1):
        def objective(w):
            n = min(len(w), len(s))
            ws = np.sum(np.array(w[:n]) * np.array(s[:n]))
            reg = np.sum(np.array(w[:n]) ** 2) * k/2
            return 0.5 * (s[-1] - ws)**2 + reg
        return objective
    
    '''
    约束矩阵、上界、下界, 这个所有评论通用
    '''
    def create_constraints(s):
        A1 = np.eye(len(s) - 1) # 不等式约束
        b1_l = np.array([0] * (len(s) - 1))
        b1_u = np.array([np.inf] * (len(s) - 1))
        constraint1 = LinearConstraint(A1, b1_l, b1_u)
        A2 = np.ones(len(s) - 1) # 等式约束
        constraint2 = LinearConstraint(A2, 1, 1)
        return [constraint1, constraint2]
    
    '''
    主逻辑
    '''
    def optimize(self, k=1):
        constraints = weight_optimizer.create_constraints(self.score_matrix[0])
        for i in range(len(self.texts)):
            s = self.score_matrix[i]
            objective = weight_optimizer.create_objective(s, k)
            result = minimize(objective, x0=np.array([0.2, 0.2, 0.2, 0.2, 0.2]), method='trust-constr', constraints=constraints)
            self.final_weight_matrix[i] = result.x
        return self.final_weight_matrix

import random

def generate_random_color():
    color_list = list(plt.cm.colors.cnames)
    return random.choice(color_list)

# if __name__ == '__main__':
#     # 准备参数
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model_save_path = './newmodel/trained_model/model_bert_wwm.pth'
#     tokenizer_save_path = './newmodel/tokenizer1e-wwm'
#     tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)
#     model = BertForSequenceClassification.from_pretrained('./newmodel/chinese_wwm_ext_pytorch', num_labels=3)
#     model.load_state_dict(torch.load(model_save_path, map_location=device))
#     model.to(device)
#     test_texts = ['虽然距离很远，但是味道不错，价格也很亲民，推荐大家可以去试一下。',
#                 '已经是第二次点了，分量真的很足，够两个人吃。\
#                 并且肉质新鲜，味道也很好，不会感觉味道重或者很有，可以说是让我吃撑了还想往嘴里炫的程度了。',
#                 '包装很环保，相比一般日常订餐有点小贵，但是东西还是不错的，非常值得，奶思极了。',
#                 '酸笋那份手抓饭的烤肉全肥的太多了，这年头实在吃不了太肥的东西了，其他都还好。',
#                 '粥太咸了，不是很推荐，速冻包子和速冻饺子还卖这么贵，而且还不好吃。']
#     star_lis = np.array([3, 3, 3, 2.4, 1.2])

#     optimizer = weight_optimizer(model, tokenizer, test_texts, star_lis, device)
#     final_weight_matrix = optimizer.optimize()

#     # 数据可视化
#     fig, axs = plt.subplots(2, 3, figsize=(10, 6))
#     aspect_lis =['environment', 'location', 'dish', 'service', 'price']
#     colors = [generate_random_color() for _ in aspect_lis]
#     for i in range(len(final_weight_matrix)):
#         sizes = [j * 100 for j in final_weight_matrix[i]]
#         row = i // 3  # 行号
#         col = i % 3   # 列号
#         axs[row, col].pie(sizes, autopct='%1.1f%%', colors=colors, shadow=True, startangle=90)
#         axs[row, col].set_title(f'comment{i}, score:{star_lis[i]}')
#     axs[1, 2].axis('off')
#     fig.legend(aspect_lis, loc='lower right')
#     # 调整子图布局以给图例留出空间
#     plt.subplots_adjust(right=0.8)
#     plt.show()