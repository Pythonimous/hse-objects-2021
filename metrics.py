import numpy as np
from sklearn.metrics import classification_report, hamming_loss, f1_score, roc_auc_score

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def multiclass_stats(y_true, y_pred):
    stats = {
    'hamming_score': hamming_score(y_true, y_pred),
    'hamming_loss': hamming_loss(y_true, y_pred),
    'f1_score': f1_score(y_true, y_pred, average='macro'),
    'roc_auc_score': roc_auc_score(y_true, y_pred, average='macro')
    }
    return classification_report(y_true, y_pred), stats


if __name__ == "__main__":

    # For comparison's sake:
    from sklearn.metrics import accuracy_score

    y_true = np.array([[0, 1, 0],
                       [0, 1, 1],
                       [1, 0, 1],
                       [0, 0, 1]])

    y_pred = np.array([[0, 1, 1],
                       [0, 1, 1],
                       [0, 1, 0],
                       [0, 0, 0]])

    print('Hamming score: {0}'.format(hamming_score(y_true, y_pred))) # 0.375 (= (0.5+1+0+0)/4)

    # Subset accuracy
    # 0.25 (= 0+1+0+0 / 4) --> 1 if the prediction for one sample fully matches the gold. 0 otherwise.
    print('Subset accuracy: {0}'.format(accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))

    # Hamming loss (smaller is better)
    # $$ \text{HammingLoss}(x_i, y_i) = \frac{1}{|D|} \sum_{i=1}^{|D|} \frac{xor(x_i, y_i)}{|L|}, $$
    # where
    #  - \\(|D|\\) is the number of samples
    #  - \\(|L|\\) is the number of labels
    #  - \\(y_i\\) is the ground truth
    #  - \\(x_i\\)  is the prediction.
    # 0.416666666667 (= (1+0+3+1) / (3*4) )
    print('Hamming loss: {0}'.format(hamming_loss(y_true, y_pred)))
