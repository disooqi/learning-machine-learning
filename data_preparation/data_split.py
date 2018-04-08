from sklearn.model_selection import train_test_split


#X should be in shape m*n where m is the number of samples and n is the number of features as well as the y_True
#the function shuffles the dataset

def split_train_dev(X, y_true, dev_size=0.25):
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, shuffle=True, test_size=dev_size)
    return X_train, X_test, y_train, y_test