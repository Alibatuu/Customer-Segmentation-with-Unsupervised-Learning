import joblib
import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Functions

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

######################################
# Görev 1: Veriyi Hazırlama
######################################

# Adım 1: flo_data_20K.csv verisini okutunuz

df_org = pd.read_csv("datasets/flo_data_20K.csv")
df = df_org.copy()
df.head()
# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.

check_df(df)

# Değişkenlerin incelenmesi

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=4, car_th=20)

for col in cat_cols:
    cat_summary(df, col)

df[num_cols].describe().T

#for col in num_cols:
#    num_summary(df, col, plot=True)

# Sayısal değişkenkerin birbirleri ile korelasyonu

correlation_matrix(df, num_cols)

for col in num_cols:
    target_summary_with_num(df, "store_type", col)

# Yanlış değişken tiplerinin değiştirilmesi

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=4, car_th=20)

# Data Preprocessing & Feature Engineering

df["order_num_total_ever"] = df["order_num_total_ever_online"] + df[
        "order_num_total_ever_offline"]
df["customer_value_total_ever"] = df["customer_value_total_ever_online"] + df[
        "customer_value_total_ever_offline"]

df["Tenure"] = (df["last_order_date"] - df["first_order_date"])
today_date = dt.datetime(2021, 6, 1)
df["Recency"] = (today_date - df["last_order_date"])
df["recency_score"] = pd.qcut(df['Recency'], 5, labels=[5, 4, 3, 2, 1])
df["frequency_score"] = pd.qcut(df["order_num_total_ever"].rank(method="first"), 5,
                                           labels=[1, 2, 3, 4, 5])
df["monetary_score"] = pd.qcut(df['customer_value_total_ever'], 5, labels=[1, 2, 3, 4, 5])
df["RFM_SCORE"] = (df['recency_score'].astype(str) +
                              df['frequency_score'].astype(str))
seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
df['segment'] = df['RFM_SCORE'].replace(seg_map, regex=True)
df.drop(["recency_score", "frequency_score", "monetary_score", "RFM_SCORE"], axis=1, inplace=True)
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=4, car_th=20)
for col in num_cols:
    print(col, check_outlier(df, col, 0.05, 0.95))
for col in num_cols:
    replace_with_thresholds(df, col)
na_col = missing_values_table(df, na_name=True)

df.drop(
        ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline", "master_id"], axis=1, inplace=True)

df["Recency"] = df["Recency"].dt.days
df["Tenure"] = df["Tenure"].dt.days
ohe_cols = [col for col in df.columns if 23 >= df[col].nunique() > 2]
# ohe_cols listesi incelendiğinde istenmeyen bazı değişkenler olduğu görülmüştür.
# Bundan dolayı cat_cols listesi incelenmiştir ve one hot encoder'dan bunların geçirilmesi
# uygun görülmüştür.
df = one_hot_encoder(df, cat_cols)
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=4, car_th=20)
cat_cols.append(cat_but_car[0]) # interested_in_categories_12 değişkeni için
df_new = one_hot_encoder(df, cat_cols, drop_first=True)
df_new.head()

#############################################
# Görev 2:  K-Means ile Müşteri Segmentasyonu
#############################################

# Adım 1: Değişkenleri standartlaştırınız.

sc = MinMaxScaler((0, 1))
df_new[num_cols] = sc.fit_transform(df_new[num_cols])

# Adım 2: Optimum küme sayısını belirleyiniz.

kmeans = KMeans()
ssd = []
K = range(1, 30)
for k in K:
    kmeans = KMeans(n_clusters=k).fit(df_new)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df_new)
elbow.elbow_value_

# Adım 3:  Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df_new)
clusters_kmeans = kmeans.labels_

df = pd.read_csv("datasets/flo_data_20K.csv", index_col=0)

df["cluster"] = clusters_kmeans
df["cluster"] = df["cluster"] + 1
df.head()


# Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.

cat_summary(df, "cluster")
df.groupby("cluster").agg(["count","mean","median"])

# flo_data_prep fonksiyonunda veri analizi, özellik mühendisliği, standartlaştırma gibi
# işlemler yapılmaktadır.

def flo_data_prep(dataframe):

    # Yanlış değişken tiplerinin değiştirilmesi
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    dataframe["order_num_total_ever_online"] = pd.to_numeric(dataframe["order_num_total_ever_online"])
    dataframe["order_num_total_ever_offline"] = pd.to_numeric(dataframe["order_num_total_ever_offline"])

    # Değişken türlerinin ayrıştırılması

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=4, car_th=20)

    # Data Preprocessing & Feature Engineering

    dataframe["order_num_total_ever"] = dataframe["order_num_total_ever_online"] + dataframe[
        "order_num_total_ever_offline"]
    dataframe["customer_value_total_ever"] = dataframe["customer_value_total_ever_online"] + dataframe[
        "customer_value_total_ever_offline"]

    dataframe["Tenure"] = (dataframe["last_order_date"] - dataframe["first_order_date"])
    today_date = dt.datetime(2021, 6, 1)
    dataframe["Recency"] = (today_date - dataframe["last_order_date"])
    dataframe["recency_score"] = pd.qcut(dataframe['Recency'], 5, labels=[5, 4, 3, 2, 1])
    dataframe["frequency_score"] = pd.qcut(dataframe["order_num_total_ever"].rank(method="first"), 5,
                                           labels=[1, 2, 3, 4, 5])
    dataframe["monetary_score"] = pd.qcut(dataframe['customer_value_total_ever'], 5, labels=[1, 2, 3, 4, 5])
    dataframe["RFM_SCORE"] = (dataframe['recency_score'].astype(str) +
                              dataframe['frequency_score'].astype(str))
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    dataframe['segment'] = dataframe['RFM_SCORE'].replace(seg_map, regex=True)
    dataframe.drop(["recency_score", "frequency_score", "monetary_score", "RFM_SCORE"], axis=1, inplace=True)
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=4, car_th=20)
    for col in num_cols:
        print(col, check_outlier(dataframe, col, 0.05, 0.95))
    for col in num_cols:
        replace_with_thresholds(dataframe, col)
    na_col = missing_values_table(dataframe, na_name=True)

    dataframe.drop(
        ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline", "master_id"
         ], axis=1, inplace=True)

    dataframe["Recency"] = dataframe["Recency"].dt.days
    dataframe["Tenure"] = dataframe["Tenure"].dt.days
    ohe_cols = [col for col in dataframe.columns if 23 >= dataframe[col].nunique() > 2]
    # ohe_cols listesi incelendiğinde istenmeyen bazı değişkenler olduğu görülmüştür.
    # Bundan dolayı cat_cols listesi incelenmiştir ve one hot encoder'dan bunların geçirilmesi
    # uygun görülmüştür.
    dataframe = one_hot_encoder(dataframe, cat_cols)
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=4, car_th=20)
    cat_cols.append(cat_but_car[0])  # interested_in_categories_12 değişkeni için
    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)
    sc = MinMaxScaler((0, 1))
    dataframe[num_cols] = sc.fit_transform(dataframe[num_cols])
    return dataframe


#############################################################
# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
#############################################################

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

# Adım 1: Görev2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.

hc_ward = linkage(df_new, "ward")

# Küme sayısının belirlenmesi

plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_ward,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=70, color='r', linestyle='--')
plt.axhline(y=65, color='b', linestyle='--')
plt.show()


# Adım 2: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=7, linkage="ward")

clusters = cluster.fit_predict(df_new)

df = pd.read_csv("datasets/flo_data_20K.csv", index_col=0)

df["hi_cluster_no"] = clusters
df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = clusters_kmeans
df["kmeans_cluster_no"] = df["kmeans_cluster_no"]  + 1
df.head()

# Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.

cat_summary(df, "hi_cluster_no")
df.groupby("hi_cluster_no").agg(["count","mean","median"])









