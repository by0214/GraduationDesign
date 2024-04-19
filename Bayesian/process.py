import pandas as pd


def process(file_path, i, train_num):
    data = pd.read_csv(file_path)

    # 定义非rating列和rating列的映射
    non_rating_mapping = {3: 'positive', 2: 'neutral', 1: 'negative'}
    rating_mapping = {5: 'five', 4: 'four', 3: 'three', 2: 'two', 1: 'one'}

    # 应用映射到非rating列
    for column in ['environment', 'location', 'dish', 'service', 'price']:
        data[column] = data[column].map(non_rating_mapping).fillna(data[column])

    # 我想让data['rating']列的每个数组都*5再/3，保留一位小数
    data['rating'] = data['rating'].apply(lambda x: round(x * 5 / 3, 1))

    # 应用映射到rating列
    data['rating'] = data['rating'].map(rating_mapping).fillna(data['rating'])

    # 保存修改后的数据到新的CSV文件
    modified_file_path = f'./Bayesian/bayesian_train_data_{train_num}_{i}.csv'
    data.to_csv(modified_file_path, index=False)

if __name__ == '__main__':
    for train_num in [5000]:
        for i in range(25):
            file_path = f'./result/user_scores_{train_num}_{i}.csv'
            process(file_path, i, train_num)
