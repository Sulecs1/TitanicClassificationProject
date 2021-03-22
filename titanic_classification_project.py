######################################################
#      TİTANİC CLASSİFİCATİON PROJECT                #
######################################################

#Gerekli Olan Kütüphaneler eklendi
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
import missingno as msno
import warnings
from sklearn.metrics import *
from sklearn.model_selection import *


from helpers.data_prep import *
from helpers.eda import *
from helpers.helpers import *



#eklentiler eklendi
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)
warnings.filterwarnings('ignore')


############################
#Veri Setini Yükleme İşlemi
#############################
df_train = pd.read_csv(r"C:\Users\Suleakcay\PycharmProjects\pythonProject6\datasets\train.csv")
df_test = pd.read_csv(r"C:\Users\Suleakcay\PycharmProjects\pythonProject6\datasets\test.csv")

titanic_train = df_train.copy()
titanic_test = df_test.copy()

titanic_train.head()
titanic_train.shape #(891, 12)

titanic_test.head()
titanic_test.shape #(418, 11)

#Aykırı değer varsa görebilmek için
msno.bar(titanic_test)
plt.show()

msno.bar(titanic_train)
plt.show()

#veri seti gözlemler hakkında inceleme yapıldı
grab_col_names(titanic_test)
grab_col_names(titanic_train)

check_df(titanic_test)
check_df(titanic_train)
#Veri setindeki eksik değerleri sorgulamak için
def df_questioning_null(df):

    print(f"Veri kümesinde hiç boş değer var mı?: {df.isnull().values.any()}")
    if df.isnull().values.any():
        null_values = df.isnull().sum()
        print(f"Hangi sütunlarda eksik değerler var?:\n{null_values[null_values > 0]}")

df_questioning_null(titanic_train)
df_questioning_null(titanic_test)


def titanic_data_prep(dataframe):

    # FEATURE ENGINEERING
    dataframe["NEW_CABIN_BOOL"] = dataframe["Cabin"].isnull().astype('int')
    dataframe["NEW_NAME_COUNT"] = dataframe["Name"].str.len()
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["Name"].apply(lambda x: len(str(x).split(" ")))
    dataframe["NEW_NAME_DR"] = dataframe["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    dataframe['NEW_TITLE'] = dataframe.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SibSp"] + dataframe["Parch"] + 1
    dataframe["NEW_AGE_PCLASS"] = dataframe["Age"] * dataframe["Pclass"]

    dataframe.loc[((dataframe['SibSp'] + dataframe['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SibSp'] + dataframe['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

    dataframe.loc[(dataframe['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['Age'] >= 18) & (dataframe['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    dataframe.loc[(dataframe['Sex'] == 'male') & (dataframe['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['Sex'] == 'male') & ((dataframe['Age'] > 21) & (dataframe['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['Sex'] == 'male') & (dataframe['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['Sex'] == 'female') & (dataframe['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['Sex'] == 'female') & ((dataframe['Age'] > 21) & (dataframe['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['Sex'] == 'female') & (dataframe['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # AYKIRI GOZLEM
    num_cols = [col for col in dataframe.columns if len(dataframe[col].unique()) > 20
                and dataframe[col].dtypes != 'O'
                and col not in "PASSENGERID"]

    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # for col in num_cols:
    #    print(col, check_outlier(df, col))
    # print(check_df(df))


    # EKSIK GOZLEM
    dataframe.drop(["TICKET", "NAME", "CABIN"], inplace=True, axis=1)
    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))

    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    # LABEL ENCODING
    binary_cols = [col for col in dataframe.columns if len(dataframe[col].unique()) == 2 and dataframe[col].dtypes == 'O']

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    dataframe = rare_encoder(dataframe, 0.01)

    ohe_cols = [col for col in dataframe.columns if 10 >= len(dataframe[col].unique()) > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)

    return dataframe


df_trainset = titanic_data_prep(titanic_train)
df_testset = titanic_data_prep(titanic_test)

################################
""" create_alone_feature(SibSp_Parch):
    if (SibSp_Parch[0] + SibSp_Parch[1]) == 0:
        return 1
    else:
        return 0


titanic_train['Alone'] = titanic_train[['SibSp', 'Parch']].apply(create_alone_feature, axis=1)
titanic_train['Familiars'] = 1 + titanic_train['SibSp'] + titanic_train['Parch']

titanic_test['Alone'] = titanic_test[['SibSp', 'Parch']].apply(create_alone_feature, axis=1)
titanic_test['Familiars'] = 1 + titanic_test['SibSp'] + titanic_test['Parch']

fig, axx = plt.subplots(2, 3, figsize=(20, 10))
axx[0, 0].set_title('Survivors')
sns.countplot(x='Survived', data=titanic_train, ax=axx[0, 0])
axx[0, 1].set_title('Survivors by Sex')
sns.countplot(x='Survived', hue='Sex', data=titanic_train, ax=axx[0, 1])
axx[0, 2].set_title('Survivors by Pclass')
sns.countplot(x='Survived', hue='Pclass', data=titanic_train, ax=axx[0, 2])
axx[1, 0].set_title('Accompanied survivors')
sns.countplot(x='Survived', hue='Alone', data=titanic_train, ax=axx[1, 0])
axx[1, 1].set_title('Accompanied survivors')
sns.countplot(x='Familiars', hue='Survived', data=titanic_train, ax=axx[1, 1])
axx[1, 2].set_title('Alone members by Pclass')
sns.countplot(x='Pclass', hue='Alone', data=titanic_train, ax=axx[1, 2])
plt.tight_layout()

create_alone_feature(titanic_test)
"""


#######################################
# Random Forests: Model & Tahmin
#######################################

y = df_trainset['SURVIVED']
X = df_trainset.drop(['SURVIVED', 'PASSENGERID'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

clf = RandomForestClassifier(n_estimators=150, random_state=42).fit(X_train, y_train)

from sklearn import metrics
y_pred = clf.predict(X_train)
print("Accuracy: {}".format(metrics.accuracy_score(y_train, y_pred)))
print(classification_report(y_train, y_pred))
#Accuracy: 0.9951845906902087

#Test hatası
y_pred = clf.predict(X_test)
print("   Accuracy: {}".format(metrics.accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))
#Accuracy: 0.8134328358208955
########################
# MODEL TUNİNG
#########################


rf_params = {"max_depth": [3, 6, 10, None],
             "max_features": [3, 5, 15],
             "n_estimators": [100, 500],
             "min_samples_split": [2, 5, 8],
             'min_samples_leaf': [1, 3, 5]}

rf_model = RandomForestClassifier(random_state=42)
rf_cv_model = GridSearchCV(clf, rf_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.get_params()
rf_cv_model.best_score_

# Model tuned
rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred)





#değişken önem düzeylerini incelemek
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(clf, X_train)

################################
# MODELİN DAHA SONRA KULLANILMAK ÜZERE KAYDEDİLMESİ
################################

import joblib
joblib.dump(clf, "rf_model_tuned_final.pkl")
rfm_model_from_disk = joblib.load("rf_model_tuned_final.pkl")
# csv formatinda kaydetmek

df = pd.DataFrame()
df["PassengerId"] = titanic_test["PASSENGERID"]
df["Survived"] = y_pred
df.to_csv("df.csv", index=False)


#zip olarak kaydetmek için
compression_opts = dict(method='zip',
                        archive_name='out.csv')
df.to_csv('out.zip', index=False,
          compression=compression_opts)
titanic_test.columns