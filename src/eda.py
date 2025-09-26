import pandas as pd

def label_distribution(df):
    return df['label'].value_counts()

def group_stats(df):
    return {
        'max_per_query': df.groupby('qid')['label'].agg(['max']).value_counts(),
        'describe_per_query': df.groupby('qid')['label'].agg(['size', 'max']).describe()
    }

if __name__ == "__main__":
    train_df = pd.read_csv('data/train.csv')
    vali_df = pd.read_csv('data/vali.csv')
    test_df = pd.read_csv('data/test.csv')

    train_counts = label_distribution(train_df)
    vali_counts = label_distribution(vali_df)
    test_counts = label_distribution(test_df)

    train_stats = group_stats(train_df)
    vali_stats = group_stats(vali_df)
    test_stats = group_stats(test_df)

    print("Train label counts:\n", train_counts)
    print("Validation label counts:\n", vali_counts)
    print("Test label counts:\n", test_counts)

    print("Train stats:\n", train_stats)
    print("Validation stats:\n", vali_stats)
    print("Test stats:\n", test_stats)
