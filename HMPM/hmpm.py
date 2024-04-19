import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import optimism as op
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class HMPM_MODEL():
    def __init__(self, weight_matrix, d=1):
        self.weight_matrix = weight_matrix
        self.T = weight_matrix.shape[0]
        self.n = weight_matrix.shape[1]
        self.d = d
        self.P_matrix = np.zeros((self.n, self.n, self.T))

    def caculate_P(self):
        for k in range(self.T):
            for j in range(self.n):
                for i in range(self.n):
                    self.P_matrix[i, j, k] = self.weight_matrix[k, i] / self.weight_matrix[k, j]
    
    def objective_function(self, x):
        # x 中前 n 个元素为 w，最后一个元素为 lambda
        return -x[-1]
    
    def constraint_sum_w(self, x):
        return np.sum(x[:-1]) - 1
    
    def constraint_positive_w(self, x):
        return x[:-1]
    
    def constraint_lambda_w_Pij(self, x):
        lmbda = x[-1]
        w = x[:-1]
        min_value = np.inf

        for i in range(self.n):
            for j in range(self.n):
                for t in range(self.T):
                    min_value = min(min_value, self.d * lmbda - (w[i] - w[j] * self.P_matrix[i, j, t]))
        return min_value - self.d

    def constraint_new_lambda_w_Pij(self, x):
        lmbda = x[-1]
        w = x[:-1]
        max_value = -np.inf

        for i in range(self.n):
            for j in range(self.n):
                for t in range(self.T):
                    max_value = max(max_value, lmbda * self.d + (w[i] - w[j] * self.P_matrix[i, j, t]))

        return self.d - max_value
    
    def optimize(self):
        self.caculate_P()
        x0 = np.array([1/self.n] * self.n + [1])

        cons = ({'type': 'eq', 'fun': self.constraint_sum_w},
        {'type': 'ineq', 'fun': self.constraint_positive_w},
        {'type': 'ineq', 'fun': self.constraint_lambda_w_Pij},
        {'type': 'ineq', 'fun': self.constraint_new_lambda_w_Pij})

        res = minimize(self.objective_function, x0, constraints=cons)
        return res.x[:-1]
    

if __name__ == '__main__':
#      # 准备参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = './newmodel/trained_model/model_bert_wwm.pth'
    tokenizer_save_path = './newmodel/tokenizer1e-wwm'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)
    model = BertForSequenceClassification.from_pretrained('./newmodel/chinese_wwm_ext_pytorch', num_labels=3)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    test_texts = ['餐厅位于北京路震庄宾馆，这家餐厅比较独特，开在宾馆内部，显得又神秘又高档，要去的话需要提前预定。我们点了一些招牌菜，比如天麻汽锅鸡，还有一个炸猪排也很好吃。考野生鲈鱼是现做现烤的，味道很鲜美。环境非常好，这个宾馆以前是国家元首居住的，在城中心，但是价格不便宜。',
                '李刚羊肉汤在环城路和吴井路交叉口，店面很小，又破又小，属于是昆明人口中的苍蝇馆子。它的清汤羊肉味道非常好，羊肉是从嵩明送来的，一公斤200左右的价格还是适中的。卫生很一般，所以我经常打包回来吃。',
                '这家餐厅位于温泉小镇，离城区比较远，原来是道班工人的住地。园区的绿化以及设施比较可以，饭菜的话以烧烤为主，口味偏咸偏油腻，十多个菜里好吃的菜只有两三个。下次可能不会再去了。',
                '茄子恰恰这家餐厅的菜是很纯正的滇味餐厅，位置离我家很近，而且菜的口味很符合我的饮食习惯，价格也不算贵。但是他家的外送情况不太好，和店里吃的不太一样。',
                '这家小锅米线是典型的昆明小吃，它的菜用的是新鲜肉，重油盐，味道很浓，吃完以后口干舌燥，感觉不是很健康。价格属于是同等店面里偏贵的，但是偶尔会想点一份吃，外卖送货很方便。']
    star_lis = np.array([2.4, 2.1, 1.2, 2.25, 1.8])
    test_texts1 = ['麒麟大口茶是新兴茶饮，只专注于一款产品，那就是糯米柠檬茶。我认为这款产品确实能够担当得起这家店的招牌产品，口味清新独特，有糯米风味和柠檬清香。价格来说不是很贵，是合理范围内的，并且量也很足，性价比是偏中上的。美中不足的是，店面只有一名员工，出餐效率偏低。',
                  '牛夫是昆明烤串专门店中数一数二的，其中蜜糖口味的肉串非常特别，让人印象深刻',
                  '状元糕类似米糕，但口感比米糕更好，并且是现点现做的模式，冬天吃一口热乎的甜糕，还能垫肚子，让人倍感幸福。此外，它价格实惠，非常亲民，如果开在我家楼下，我很可能经常光顾',
                  '意老夫子是一家西餐店，我认为它的烤肠比较一般，是无功无过的水平；大蒜酱烤面包片非常难吃，而且性价比很低，四片就卖二十五非常得坑；炸鱼薯条比较好吃，但薯条也没有那么好吃，像是半成品',
                  '火瓢牛肉是一家清真馆，离我家比较近，开车只需要五分钟。它家口味重麻重辣，吃起来很过瘾，并且牛肉品质很好，价格也不贵。如果说还有提升之处，那我认为就是菜品数量还可以再丰富一些']
    star_lis1 = np.array([2.12, 2.78, 2.17, 1.98, 2.82])

    optimizer = op.weight_optimizer(model, tokenizer, test_texts, star_lis, device)
    final_weight_matrix = optimizer.optimize()

    hmpm_model = HMPM_MODEL(final_weight_matrix)
    preference = hmpm_model.optimize()

    # 数据可视化
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    aspect_lis =['environment', 'location', 'dish', 'service', 'price']
    colors = [op.generate_random_color() for _ in aspect_lis]
    for i in range(len(final_weight_matrix)):
        sizes = [j * 100 for j in final_weight_matrix[i]]
        row = i // 3  # 行号
        col = i % 3   # 列号
        axs[row, col].pie(sizes, autopct='%1.1f%%', colors=colors, shadow=True, startangle=90)
        axs[row, col].set_title(f'comment{i}, score:{star_lis[i]}')
    axs[1, 2].pie(preference * 100, autopct='%1.1f%%', colors=colors, shadow=True, startangle=90)
    axs[1, 2].set_title('preference')
    fig.legend(aspect_lis, loc='lower right')
    # 调整子图布局以给图例留出空间
    plt.subplots_adjust(right=0.8)
    plt.show()


