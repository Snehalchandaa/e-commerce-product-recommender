import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# LOAD DATA
data = pd.read_csv("transactions.csv")
transactions = data['Transaction'].apply(lambda x: x.split(',')).tolist()

# ENCODE DATA
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_data, columns=te.columns_)

# APPLY APRIORI
frequent_items = apriori(df, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.3)

# RECOMMEND FUNCTION
def recommend(products):
    recommendations = []


    for i in range(len(rules)):
        antecedents = list(rules.iloc[i]['antecedents'])
        consequents = list(rules.iloc[i]['consequents'])
        confidence = rules.iloc[i]['confidence']

        if any(item in antecedents for item in products):
            for item in consequents:
                if item not in products:
                    recommendations.append((item, confidence))

    
    if not recommendations:
        for t in transactions:
            if any(item in t for item in products):
                for item in t:
                    if item not in products:
                        recommendations.append((item, 0.1))  

    # Remove duplicates
    unique = {}
    for item, conf in recommendations:
        if item not in unique or conf > unique[item]:
            unique[item] = conf

    return sorted(unique.items(), key=lambda x: x[1], reverse=True)

# STREAMLIT
st.title("🛒 Smart Product Recommender")

product_list = sorted(df.columns.tolist())
selected_products = st.multiselect("Select products:", product_list)

# REAL-TIME RECOMMENDATION
if selected_products:
    results = recommend(selected_products)

    st.subheader("Customers who bought this also bought:")

    for item, conf in results:
        st.write(f"👉 {item} (confidence: {conf:.2f})")

# 📊 GRAPH 1: TOP PRODUCTS
st.subheader("📊 Most Purchased Products")

product_counts = df.sum().sort_values(ascending=False)

fig1, ax1 = plt.subplots()
ax1.bar(product_counts.index, product_counts.values)
plt.xticks(rotation=45)

st.pyplot(fig1)

# 📊 GRAPH 2: SELECTED PRODUCT RELATION
if selected_products:
    st.subheader("📊 Related Products Frequency")

    related_counts = {}

    for t in transactions:
        if any(item in t for item in selected_products):
            for item in t:
                if item not in selected_products:
                    related_counts[item] = related_counts.get(item, 0) + 1

    if related_counts:
        fig2, ax2 = plt.subplots()
        ax2.bar(related_counts.keys(), related_counts.values())
        plt.xticks(rotation=45)

        st.pyplot(fig2)