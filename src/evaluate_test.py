import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from model_dataset import LambdaRankDataset, LambdaRankModel
from preprocessing import load_letor_data


def evaluate_model(df, model):
    X = df.drop(['label', 'qid'], axis=1)
    y = df['label']
    preds = model.model.predict(X)
    qid = df['qid']

    results_df = pd.DataFrame({'qid': qid, 'true_label': y, 'pred_score': preds})
    results_list = []

    for qid, group in results_df.groupby('qid'):
        y_true = group['true_label'].values
        y_pred = group['pred_score'].values
        num_docs = len(y_true)
        if num_docs > 1:
            try:
                ndcg_5 = ndcg_score([y_true], [y_pred], k=5)
                ndcg_10 = ndcg_score([y_true], [y_pred], k=10)
            except:
                ndcg_5, ndcg_10 = np.nan, np.nan
        else:
            ndcg_5, ndcg_10 = np.nan, np.nan

        results_list.append({
            'qid': qid,
            'num_docs': num_docs,
            'max_label': np.max(y_true),
            'min_label': np.min(y_true),
            'sum_relevant': np.sum(y_true > 0),
            'ndcg@5': ndcg_5,
            'ndcg@10': ndcg_10
        })

    return pd.DataFrame(results_list)


if __name__ == "__main__":
    fold_path = "/kaggle/input/letor4/MQ2008/Fold1"
    test_df = load_letor_data(fold_path + "/test.txt")

    # Предполагаем, что модель уже обучена и сохранена
    # Для примера создаём модель и загружаем train/vali, чтобы обучить
    train_df = load_letor_data(fold_path + "/train.txt")
    vali_df = load_letor_data(fold_path + "/vali.txt")
    train_dataset = LambdaRankDataset(train_df)
    vali_dataset = LambdaRankDataset(vali_df, reference=train_dataset.dataset)
    model = LambdaRankModel()
    model.train(train_dataset, vali_dataset)

    test_results_df = evaluate_model(test_df, model)

    final_ndcg_5_test = test_results_df[test_results_df['max_label'] > 0]['ndcg@5'].mean()
    final_ndcg_10_test = test_results_df[test_results_df['max_label'] > 0]['ndcg@10'].mean()

    print("=== ТЕСТ ===")
    print(f"Всего запросов: {len(test_results_df)}")
    print(f"Финальный NDCG@5 (релевантные запросы): {final_ndcg_5_test:.4f}")
    print(f"Финальный NDCG@10 (релевантные запросы): {final_ndcg_10_test:.4f}")
