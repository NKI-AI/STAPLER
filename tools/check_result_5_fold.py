from sklearn.metrics import average_precision_score
import json
import pandas as pd

df_test = pd.read_csv('vdjdb_external_negatives_data.csv')
df_train = pd.read_csv('2022-05-20_train-set_full-seq.csv')
train_epitopes = df_train['epitope_aa'].unique().tolist()

results_fold_dict = {}
pred_cls_mean = [0] * len(df_test)
for i in range(5):
    with open(f'/tensorboard/version_{i}/test_results{i}.json') as f:
        data = json.load(f)
        pred_cls = []
        label_cls = []
        for batch in data:
            pred_cls.append(batch['preds_cls'])
            label_cls.append(batch['labels_cls'])

        pred_cls = [item for sublist in pred_cls for item in sublist]
        label_cls = [item for sublist in label_cls for item in sublist]
        # assert labels are same in test and fold
        assert df_test['label_true_pair'].tolist() == label_cls
        pred_cls_mean = [x + y / 5 for x, y in zip(pred_cls_mean, pred_cls)]

df_test['pred_cls'] = pred_cls_mean
df_test['in_train'] = df_test['epitope_aa'].isin(train_epitopes)

df_test_in_train = df_test[df_test['in_train'] == True]
df_test_not_in_train = df_test[df_test['in_train'] == False]

# AP seen
average_precision_seen = average_precision_score(y_true=df_test_in_train['label_true_pair'],
                                                 y_score=df_test_in_train['pred_cls'])
print('Average precision-recall score seen: {0:0.5f}'.format(average_precision_seen))

# AP unseen
average_precision_unseen = average_precision_score(y_true=df_test_not_in_train['label_true_pair'],
                                                   y_score=df_test_not_in_train['pred_cls'])
print('Average precision-recall score unseen: {0:0.5f}'.format(average_precision_unseen))
