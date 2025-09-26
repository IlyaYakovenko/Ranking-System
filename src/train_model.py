import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from model_dataset import LambdaRankDataset, LambdaRankModel
from preprocessing import load_letor_data

def run_training_pipeline(fold_path):
    train_df = load_letor_data(fold_path + "/train.txt")
    vali_df = load_letor_data(fold_path + "/vali.txt")

    train_dataset = LambdaRankDataset(train_df)
    vali_dataset = LambdaRankDataset(vali_df, reference=train_dataset.dataset)

    model = LambdaRankModel()
    model.train(train_dataset, vali_dataset)

    X_vali = vali_df.drop(['label', 'qid'], axis=1)
    y_vali = vali_df['label']
    preds_vali = model.model.predict(X_vali)
    qid_vali = vali_df['qid']

    vali_results_df = pd.DataFrame({
        'qid': qid_vali,
        'true_label': y_vali,
        'pred_score': preds_vali
    })

    results_list = []
    for qid, group in vali_results_df.groupby('qid'):
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

    query_results_df = pd.DataFrame(results_list)

    print("=== ОБЩАЯ СТАТИСТИКА ПО ВАЛИДАЦИИ ===")
    print(f"Всего запросов: {len(query_results_df)}")
    print(f"Vali NDCG@5: {query_results_df['ndcg@5'].mean():.4f}")
    print(f"Vali NDCG@10: {query_results_df['ndcg@10'].mean():.4f}")

    return model, query_results_df

if __name__ == "__main__":
    fold_path = "/kaggle/input/letor4/MQ2008/Fold1"
    run_training_pipeline(fold_path)
