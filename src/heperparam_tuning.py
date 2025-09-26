import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import ndcg_score
from model_dataset import LambdaRankDataset, LambdaRankModel
from preprocessing import load_letor_data

fold_path = "/kaggle/input/letor4/MQ2008/Fold1"

train_df = load_letor_data(fold_path + "/train.txt")
vali_df = load_letor_data(fold_path + "/vali.txt")
test_df = load_letor_data(fold_path + "/test.txt")

x_train = train_df.drop(['label', 'qid'], axis=1)
y_train = train_df['label']
qid_train = train_df['qid']

x_vali = vali_df.drop(['label', 'qid'], axis=1)
y_vali = vali_df['label']
qid_vali = vali_df['qid']

x_test = test_df.drop(['label', 'qid'], axis=1)
y_test = test_df['label']

def create_datasets(feature_pre_filter=True):
    train_data = LambdaRankDataset(train_df)
    vali_data = LambdaRankDataset(vali_df, reference=train_data.dataset)
    return train_data, vali_data

def compute_fair_ndcg(y_true, y_pred, qids, k=5):
    ndcg_scores = []
    for qid in np.unique(qids):
        mask = qids == qid
        q_true = y_true[mask]
        q_pred = y_pred[mask]
        if len(q_true) > 1 and np.max(q_true) > 0:
            try:
                ndcg = ndcg_score([q_true], [q_pred], k=k)
                if not np.isnan(ndcg):
                    ndcg_scores.append(ndcg)
            except:
                continue
    return np.mean(ndcg_scores) if ndcg_scores else 0

def objective(trial):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [5],
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.7, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
        'early_stopping_rounds': 20
    }

    train_data_optuna, vali_data_optuna = create_datasets(feature_pre_filter=False)

    model = LambdaRankModel(params=params)
    model.train(train_data_optuna, vali_data_optuna)

    vali_preds = model.model.predict(x_vali, num_iteration=model.model.best_iteration)
    return 1 - compute_fair_ndcg(y_vali.values, vali_preds, qid_vali.values, k=5)

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    print("Запуск оптимизации гиперпараметров...")
    study.optimize(objective, n_trials=500, show_progress_bar=True)

    print("\n=== ЛУЧШИЕ ГИПЕРПАРАМЕТРЫ ===")
    print(f"Лучшее значение (1 - NDCG@5): {study.best_value:.4f}")
    print(f"Лучший NDCG@5: {1 - study.best_value:.4f}")
    print(f"Лучшие параметры: {study.best_params}")

    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [5, 10],
        'verbose': 1,
        'early_stopping_rounds': 50
    })

    train_data_final, vali_data_final = create_datasets(feature_pre_filter=False)

    final_model = LambdaRankModel(params=best_params)
    final_model.train(train_data_final, vali_data_final)

    vali_preds_final = final_model.model.predict(x_vali)
    final_ndcg_5_vali = compute_fair_ndcg(y_vali.values, vali_preds_final, qid_vali.values, 5)
    final_ndcg_10_vali = compute_fair_ndcg(y_vali.values, vali_preds_final, qid_vali.values, 10)

    test_preds_final = final_model.model.predict(x_test)
    tuned_ndcg_5_test = compute_fair_ndcg(y_test.values, test_preds_final, test_df['qid'].values, 5)
    tuned_ndcg_10_test = compute_fair_ndcg(y_test.values, test_preds_final, test_df['qid'].values, 10)

    print(f"\n=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ===")
    print(f"Vali NDCG@5: {final_ndcg_5_vali:.4f}")
    print(f"Vali NDCG@10: {final_ndcg_10_vali:.4f}")
    print(f"Test NDCG@5: {tuned_ndcg_5_test:.4f}")
    print(f"Test NDCG@10: {tuned_ndcg_10_test:.4f}")

    print(f"\n=== СРАВНЕНИЕ С BASELINE ===")
    print(f"Улучшение Test NDCG@5: {tuned_ndcg_5_test - final_ndcg_5_vali:.4f}")
    print(f"Улучшение Test NDCG@10: {tuned_ndcg_10_test - final_ndcg_10_vali:.4f}")
