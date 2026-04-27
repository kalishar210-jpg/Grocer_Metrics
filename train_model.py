import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_excel('Online_Retail.xlsx')
basket = data.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack(fill_value=0)
basket = basket.apply(lambda x: x.map(lambda y: 1 if y > 0 else 0))
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.to_csv('recommender_model.csv', index=False)

print("Recommender model saved → recommender_model.csv")