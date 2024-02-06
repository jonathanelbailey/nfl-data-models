from sklearn.model_selection import GroupKFold


def train_test_split(X, y, groups, n_splits):
    group_fold = GroupKFold(n_splits=n_splits)

    for train_index, test_index in group_fold.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, y_train, X_test, y_test


def get_groups_by_index(X, groups):
    return groups[X.index]