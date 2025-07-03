# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="Consumer Marketing Analytics Dashboard", layout="wide")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

if 'uploaded_data' not in st.session_state:
    df = load_data('synthetic_consumer_marketing_survey.csv')
else:
    df = st.session_state['uploaded_data']

st.sidebar.title("Upload Data")
up_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
if up_file:
    df = pd.read_csv(up_file)
    st.session_state['uploaded_data'] = df
    st.sidebar.success("Uploaded!")

st.sidebar.download_button("Download Data", df.to_csv(index=False), "consumer_data.csv", "text/csv")

tabs = st.tabs(["Data Visualization","Classification","Clustering","Association Rule Mining","Regression"])

# ---- Data Visualization ----
with tabs[0]:
    st.header("Descriptive Insights")
    st.subheader("Age Group Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Age_Group", data=df, order=df['Age_Group'].value_counts().index, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Income Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(x=df['Income'], ax=ax, showfliers=True)
    st.pyplot(fig)

    st.subheader("Marketing Channel Popularity")
    channel_counts = df['Marketing_Channels'].str.get_dummies(sep=', ').sum().sort_values(ascending=False)
    st.bar_chart(channel_counts)

# ---- Classification ----
with tabs[1]:
    st.header("Classification")
    target = "Purchase_Last_3mo"
    features = ['Age_Group','Gender','Education','Region','Social_Media_Time','Products_Purchased',
                'Likelihood_Try_New_Brand','Referral_Willingness','Comfort_Data_Personalization',
                'Cart_Abandon_Freq','Tech_Attitude']
    clf_df = df.dropna(subset=[target])
    X = pd.get_dummies(clf_df[features], drop_first=True)
    y = clf_df[target].map({'Yes':1,'No':0})
    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.3,random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8),
        "GBRT": GradientBoostingClassifier(random_state=42)
    }
    results=[]
    for n,m in models.items():
        m.fit(X_train,y_train)
        y_pred = m.predict(X_test)
        results.append({"Model":n,
                        "Train Acc":accuracy_score(y_train,m.predict(X_train)),
                        "Test Acc":accuracy_score(y_test,y_pred),
                        "Precision":precision_score(y_test,y_pred),
                        "Recall":recall_score(y_test,y_pred),
                        "F1":f1_score(y_test,y_pred)})
    st.dataframe(pd.DataFrame(results).round(3))

    st.subheader("Confusion Matrix")
    choice = st.selectbox("Model", list(models.keys()))
    sel_model = models[choice]
    cm = confusion_matrix(y_test, sel_model.predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=['No','Yes'], yticklabels=['No','Yes'])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for n,m in models.items():
        if hasattr(m,"predict_proba"):
            proba = m.predict_proba(X_test)[:,1]
        else:
            proba = m.decision_function(X_test)
        fpr,tpr,_ = roc_curve(y_test, proba)
        ax.plot(fpr,tpr,label=f"{n}")
    ax.plot([0,1],[0,1],'k--'); ax.legend(); st.pyplot(fig)

# ---- Clustering ----
with tabs[2]:
    st.header("Clustering")
    clust_feats = ['Age_Group','Gender','Education','Region','Social_Media_Time','Products_Purchased',
                   'Likelihood_Try_New_Brand','Referral_Willingness','Comfort_Data_Personalization',
                   'Cart_Abandon_Freq','Tech_Attitude']
    clust_df = df[clust_feats].dropna()
    Xc = pd.get_dummies(clust_df, drop_first=True)
    k = st.slider("Clusters",2,10,4)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xc)
    clust_df['Cluster'] = labels
    st.dataframe(clust_df.groupby('Cluster').agg(lambda x: x.value_counts().index[0]))

    st.download_button("Download Clustered", clust_df.to_csv(index=False), "clustered.csv","text/csv")

    inertia=[]
    for i in range(2,11):
        inertia.append(KMeans(n_clusters=i, random_state=42, n_init=10).fit(Xc).inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(2,11), inertia, marker="o")
    ax.set_xlabel("Clusters"); ax.set_ylabel("Inertia"); st.pyplot(fig)

# ---- Association Rule Mining ----
with tabs[3]:
    st.header("Association Rule Mining")
    multi_cols=['Marketing_Channels','Influencing_Factors','Biggest_Challenge']
    col1 = st.selectbox("Column 1", multi_cols)
    col2 = st.selectbox("Column 2", multi_cols, index=1)
    sup = st.slider("Min Support",0.01,0.5,0.05)
    conf = st.slider("Min Confidence",0.1,1.0,0.5)
    lift_th = st.slider("Min Lift",0.5,3.0,1.0)

    trans=[]
    for _,row in df[[col1,col2]].dropna().iterrows():
        items=set()
        items.update([i.strip() for i in str(row[col1]).split(",")])
        items.update([i.strip() for i in str(row[col2]).split(",")])
        trans.append(list(items))
    te = TransactionEncoder()
    te_ary = te.fit(trans).transform(trans)
    trans_df = pd.DataFrame(te_ary, columns=te.columns_)
    freq = apriori(trans_df, min_support=sup, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=conf)
    rules = rules[rules['lift']>=lift_th].sort_values('confidence', ascending=False).head(10)
    st.write(rules[['antecedents','consequents','support','confidence','lift']])

# ---- Regression ----
with tabs[4]:
    st.header("Regression")
    reg_target = st.selectbox("Target", ["Income","Monthly_Online_Spend"], index=1)
    reg_feats = ['Age_Group','Gender','Education','Region','Social_Media_Time','Products_Purchased',
                 'Likelihood_Try_New_Brand','Referral_Willingness','Comfort_Data_Personalization',
                 'Cart_Abandon_Freq','Tech_Attitude']
    reg_df = df.dropna(subset=[reg_target])
    Xr = pd.get_dummies(reg_df[reg_feats], drop_first=True)
    yr = reg_df[reg_target]
    X_train,X_test,y_train,y_test = train_test_split(Xr,yr,test_size=0.3,random_state=42)

    regs = {"Linear":LinearRegression(),
            "Ridge":Ridge(alpha=1.0),
            "Lasso":Lasso(alpha=0.5),
            "DecisionTree":DecisionTreeRegressor(max_depth=6, random_state=42)}
    res=[]
    for n,r in regs.items():
        r.fit(X_train,y_train)
        pred=r.predict(X_test)
        res.append({"Model":n,
                    "R2":r.score(X_test,y_test),
                    "RMSE":np.sqrt(np.mean((y_test-pred)**2))})
    st.dataframe(pd.DataFrame(res).round(2))
