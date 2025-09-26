import os
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from preprocessing import load_letor_data
from model_dataset import Dataset, train
from hyperopt_tuning import best_params

def train_on_all_data_and_test_on_combined():
    def train_on_all_data_and_test_on_combined():
        """Train on all training+validation data and test on combined test set"""

        all_train_dfs = []
        all_test_dfs = []

        print("Loading data from all folds...")

        # 1. Собираем все тренировочные и тестовые данные
        for fold_id in range(1, 6):
            fold_path = f"/kaggle/input/letor4/MQ2008/Fold{fold_id}"

            # Загружаем train + vali для обучения
            train_df = load_letor_data(os.path.join(fold_path, "train.txt"))
            vali_df = load_letor_data(os.path.join(fold_path, "vali.txt"))

            # Объединяем train + vali
            combined_train_df = pd.concat([train_df, vali_df], ignore_index=True)
            all_train_dfs.append(combined_train_df)

            # Тестовые данные для объединения
            test_df = load_letor_data(os.path.join(fold_path, "test.txt"))
            all_test_dfs.append(test_df)

        # 2. Объединяем все тренировочные данные
        final_train_df = pd.concat(all_train_dfs, ignore_index=True)

        # 3. Объединяем все тестовые данные в один большой набор
        final_test_df = pd.concat(all_test_dfs, ignore_index=True)
        print(f"Объединенный тестовый набор: {len(final_test_df)} записей")
        print(f"Уникальных запросов в тесте: {final_test_df['qid'].nunique()}")

        # 4. Подготовка данных для обучения
        X_train_all = final_train_df.drop(['label', 'qid'], axis=1)
        y_train_all = final_train_df['label']
        group_train_all = final_train_df.groupby('qid').size().tolist()

        train_data_all = Dataset(X_train_all, label=y_train_all, group=group_train_all)

        # 5. Обучаем финальную модель на ВСЕХ данных
        print("Training final model on all data...")

        params = best_params
        print(params)

        final_model = train(
            params,
            train_data_all,
            num_boost_round=50
        )

        # 6. Тестируем на ОБЪЕДИНЕННОМ тестовом наборе
        print("Testing on combined test set...")
        X_test_combined = final_test_df.drop(['label', 'qid'], axis=1)
        y_test_combined = final_test_df['label']
        qid_test_combined = final_test_df['qid']

        test_preds_combined = final_model.predict(X_test_combined)

        # 7. ДЕТАЛЬНЫЙ АНАЛИЗ на объединенном тестовом наборе
        test_results_df = pd.DataFrame({
            'qid': qid_test_combined,
            'true_label': y_test_combined,
            'pred_score': test_preds_combined
        })

        results_list = []

        for qid, group in test_results_df.groupby('qid'):
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

            max_label = np.max(y_true)
            min_label = np.min(y_true)
            sum_relevant = np.sum(y_true > 0)

            results_list.append({
                'qid': qid,
                'num_docs': num_docs,
                'max_label': max_label,
                'min_label': min_label,
                'sum_relevant': sum_relevant,
                'ndcg@5': ndcg_5,
                'ndcg@10': ndcg_10
            })

        query_results_df = pd.DataFrame(results_list)

        print("\n" + "=" * 60)
        print("ДЕТАЛЬНЫЙ АНАЛИЗ НА ОБЪЕДИНЕННОМ ТЕСТОВОМ НАБОРЕ")
        print("=" * 60)

        print(f"Всего запросов: {len(query_results_df)}")
        print(f"Всего документов: {len(final_test_df)}")
        print()

        print("=== ОБЩАЯ СТАТИСТИКА ===")
        print(f"Test NDCG@5 (все запросы): {query_results_df['ndcg@5'].mean():.4f}")
        print(f"Test NDCG@10 (все запросы): {query_results_df['ndcg@10'].mean():.4f}")
        print()

        print("=== РАСПРЕДЕЛЕНИЕ NDCG ПО ЗАПРОСАМ ===")
        print("NDCG@5:")
        print(query_results_df['ndcg@5'].describe())
        print("\nNDCG@10:")
        print(query_results_df['ndcg@10'].describe())
        print()

        print("=== АНАЛИЗ ПО КАЧЕСТВУ ЗАПРОСОВ ===")
        print("\nЗапросы БЕЗ релевантных документов (max_label = 0):")
        no_relevant = query_results_df[query_results_df['max_label'] == 0]
        print(f"Количество: {len(no_relevant)}")
        print(f"Средний NDCG@5: {no_relevant['ndcg@5'].mean():.4f}")
        print(f"Средний NDCG@10: {no_relevant['ndcg@10'].mean():.4f}")

        print("\nЗапросы С релевантными документами (max_label > 0):")
        has_relevant = query_results_df[query_results_df['max_label'] > 0]
        print(f"Количество: {len(has_relevant)}")
        print(f"Средний NDCG@5: {has_relevant['ndcg@5'].mean():.4f}")
        print(f"Средний NDCG@10: {has_relevant['ndcg@10'].mean():.4f}")

        print("\nЗапросы с высокорелевантными документами (max_label = 2):")
        has_high_relevant = query_results_df[query_results_df['max_label'] == 2]
        print(f"Количество: {len(has_high_relevant)}")
        print(f"Средний NDCG@5: {has_high_relevant['ndcg@5'].mean():.4f}")
        print(f"Средний NDCG@10: {has_high_relevant['ndcg@10'].mean():.4f}")

        # 8. ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (только релевантные запросы)
        final_ndcg_5_test = query_results_df[query_results_df['max_label'] > 0]['ndcg@5'].mean()
        final_ndcg_10_test = query_results_df[query_results_df['max_label'] > 0]['ndcg@10'].mean()

        print("\n" + "=" * 60)
        print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 60)
        print(f"Финальный NDCG@5 на тесте (по релевантным запросам): {final_ndcg_5_test:.4f}")
        print(f"Финальный NDCG@10 на тесте (по релевантным запросам): {final_ndcg_10_test:.4f}")

        return final_model, query_results_df

    # Запускаем
    final_model, combined_test_results = train_on_all_data_and_test_on_combined()

    # Сохраняем финальную модель
    final_model.save_model('final_ranking_model.txt')
    print("Final model saved to 'final_ranking_model.txt'")

    # Сохраняем результаты анализа
    combined_test_results.to_csv('combined_test_analysis.csv', index=False)
    print("Test analysis saved to 'combined_test_analysis.csv'")


if __name__ == "__main__":
    final_model, combined_test_results = train_on_all_data_and_test_on_combined()
    final_model.save_model('final_ranking_model.txt')
    combined_test_results.to_csv('combined_test_analysis.csv', index=False)
