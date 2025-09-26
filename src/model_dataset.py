import pandas as pd
import lightgbm as lgb

class LambdaRankDataset:
    def __init__(self, df, reference=None):
        self.df = df.sort_values(by="qid")
        self.group = self.df.groupby("qid").size().tolist()
        self.X = self.df.drop(['label', 'qid'], axis=1)
        self.y = self.df['label']
        self.dataset = lgb.Dataset(self.X, label=self.y, group=self.group, reference=reference)

class LambdaRankModel:
    def __init__(self, params=None, num_boost_round=500):
        if params is None:
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'eval_at': [5, 10],
                'learning_rate': 0.05,
                'verbose': 1,
                'early_stopping_rounds': 50
            }
        self.params = params
        self.num_boost_round = num_boost_round
        self.model = None

    def train(self, train_dataset, vali_dataset):
        self.model = lgb.train(
            self.params,
            train_dataset.dataset,
            valid_sets=[vali_dataset.dataset],
            valid_names=['vali'],
            num_boost_round=self.num_boost_round
        )
        return self.model
