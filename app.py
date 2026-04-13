import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

# =========================
# إعداد الصفحة
# =========================
st.set_page_config(page_title="Retail Dashboard", layout="wide")

st.title("🛒 Online Retail Dashboard")
st.markdown("### Data Analysis & Customer Segmentation")

# =========================
# تحميل البيانات
# =========================
@st.cache_data
def load_data():
    data = pd.read_excel("Online Retail (1).xlsx")

    # Cleaning
    data = data.drop_duplicates()
    data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
    data = data.dropna(subset=['CustomerID'])

    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

    # Features
    data['Year'] = data['InvoiceDate'].dt.year
    data['Month'] = data['InvoiceDate'].dt.month
    data['Day'] = data['InvoiceDate'].dt.day
    data['Hour'] = data['InvoiceDate'].dt.hour

    data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

    return data

data = load_data()

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Filters")

country = st.sidebar.selectbox("Select Country", ["All"] + list(data['Country'].unique()))

if country != "All":
    data = data[data['Country'] == country]

# =========================
# KPIs
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Revenue", f"{data['TotalPrice'].sum():,.0f}")
col2.metric("🧾 Total Orders", data['InvoiceNo'].nunique())
col3.metric("👥 Customers", data['CustomerID'].nunique())

# =========================
# Data Preview
# =========================
st.subheader("📄 Data Preview")
st.dataframe(data.head())

# =========================
# Top Products
# =========================
st.subheader("🔥 Top 10 Products")

top_products = (
    data.groupby('Description')['Quantity']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig1 = px.bar(top_products, x='Quantity', y='Description', orientation='h',
              title="Top Products by Quantity")

st.plotly_chart(fig1, use_container_width=True)

# =========================
# Revenue by Country
# =========================
st.subheader("🌍 Revenue by Country")

country_sales = data.groupby('Country')['TotalPrice'].sum().reset_index()

fig2 = px.bar(country_sales, x='Country', y='TotalPrice',
              title="Revenue per Country")

st.plotly_chart(fig2, use_container_width=True)

# =========================
# Sales by Hour
# =========================
st.subheader("⏰ Sales by Hour")

sales_hour = data.groupby('Hour')['TotalPrice'].sum().reset_index()

fig3 = px.line(sales_hour, x='Hour', y='TotalPrice',
               title="Sales Distribution Over Hours")

st.plotly_chart(fig3, use_container_width=True)

# =========================
# RFM Analysis
# =========================
st.subheader("🧠 Customer Segmentation (RFM)")

snapshot_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

# اختيار عدد الكلاستر
k = st.slider("Select number of clusters", 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm)

# عرض البيانات
st.dataframe(rfm.head())

# =========================
# Visualization Clusters
# =========================
fig4 = px.scatter(rfm, x='Recency', y='Monetary',
                  color=rfm['Cluster'].astype(str),
                  title="Customer Segments")

st.plotly_chart(fig4, use_container_width=True)

# =========================
# Prediction Section
# =========================
st.subheader("🎯 Predict Customer Segment")

col1, col2, col3 = st.columns(3)

recency = col1.number_input("Recency", 0, 1000, 100)
frequency = col2.number_input("Frequency", 0, 1000, 10)
monetary = col3.number_input("Monetary", 0, 100000, 500)

if st.button("Predict Cluster"):
    user_data = np.array([[recency, frequency, monetary]])
    cluster = kmeans.predict(user_data)

    st.success(f"✅ Customer belongs to Cluster: {cluster[0]}")