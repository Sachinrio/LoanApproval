import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold,cross_val_score
l=LabelEncoder()
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title("📊 Loan Approval Prediction ")
data=pd.read_csv("Loan1.csv")

st.subheader("DataPreview")
col=data.select_dtypes(include="object").columns.to_list()
print(col)
st.dataframe(data.head())

for i in col:
    data[i]=l.fit_transform(data[i])
print(data.head())
print(data["Employment_Status"].nunique())
x=data.drop("Approval",axis=1)
y=data["Approval"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
st.subheader("Select the algorithm")
model=st.selectbox("",options=[LogisticRegression(max_iter=3000),DecisionTreeClassifier(),RandomForestClassifier(n_estimators=100),KNeighborsClassifier(n_neighbors=15),AdaBoostClassifier(random_state=42),XGBClassifier(random_state=42)])

st.write(model)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
st.write("Accuracy",accuracy_score(y_test,y_pred))
st.write("Precision_score",precision_score(y_test,y_pred))
st.write("recall_score",recall_score(y_test,y_pred))
st.write("confusion_matrix",confusion_matrix(y_test,y_pred))
st.write("Classification_report",classification_report(y_test,y_pred))
all_columns = data.columns.tolist()

st.sidebar.header("📈 Visualization Settings")
chart_type = st.sidebar.selectbox("Select Chart Type", 
                                      ["Line Chart", "Bar Chart", "Scatter Plot", 
                                       "Pie Chart", "Histogram", "Heatmap"])
    
x_axis = st.sidebar.selectbox("X-Axis", options=all_columns)
y_axis = st.sidebar.selectbox("Y-Axis", options=all_columns)

# Generate chart
st.subheader(f"{chart_type}")
if chart_type == "Line Chart":
    fig = px.line(data, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Bar Chart":
    fig = px.bar(data, x=x_axis, y=y_axis, color=x_axis, title=f"{y_axis} by {x_axis}")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Scatter Plot":
    fig = px.scatter(data, x=x_axis, y=y_axis, color=x_axis, title=f"{y_axis} vs {x_axis}")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Pie Chart":
    fig = px.pie(data, names=x_axis, values=y_axis, title=f"{x_axis} distribution")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Histogram":
    fig = px.histogram(data, x=x_axis, nbins=20, title=f"Distribution of {x_axis}")
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Heatmap":
    corr = data[[x_axis,y_axis]].corr()
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
        
    sns.heatmap(corr, annot=True, cmap='coolwarm',ax=ax)
    st.pyplot(fig)
st.subheader("🔍 Predict Loan Approval for New Customer")
mapping={"Employed":0,"Unemployed":1}
datas=[]
print(data.columns)
with st.form("myForm"):
   for i in x.columns:
        if i not in col:
           a=st.number_input(f"Enter the {i} value")
           datas.append(a)
        else:
           a=st.selectbox(f"Enter the {i} value",options=["Employed","Unemployed"])
           datas.append(mapping[a])
   sub=st.form_submit_button("predict")
mapping={"Employed":0,"Unemployed":1}
if sub:
    model.fit(x_train,y_train)
    input_data=np.array(datas).reshape(1,-1)
    y_pred=model.predict(input_data)
    if y_pred== 0:
        st.success("Loan Approved ✔")
    else:
        st.error("Loan Rejected ❌")
