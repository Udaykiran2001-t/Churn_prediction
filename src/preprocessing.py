import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
def load_data(path):
    df = pd.read_csv(path, index_col=0)
    df.columns = map(str.lower, df.columns)
    return df

def encode_features(df):
    df = pd.get_dummies(df, columns=["geography", "gender"], drop_first=True)
    df.drop(["customerid", "surname"], axis=1, inplace=True)
    return df

def scale_features(X):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

def split_and_balance(X, y, test_size=0.2, random_state=12345):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    smk = SMOTETomek(random_state=random_state)
    X_train, y_train = smk.fit_resample(X_train, y_train)
    X_test, y_test = smk.fit_resample(X_test, y_test)
    return X_train, X_test, y_train, y_test
