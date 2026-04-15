print("Madhusri S-24BAD065")

# 1. Import libraries
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# 2. Load dataset (sample if file not available)
transactions = [
    ['milk', 'bread', 'butter'],
    ['beer', 'bread'],
    ['milk', 'bread', 'butter', 'beer'],
    ['bread', 'butter'],
    ['milk', 'bread', 'beer'],
    ['milk', 'butter'],
    ['bread', 'butter'],
    ['milk', 'bread', 'butter', 'beer']
]
# 3. One-hot encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

print("\nOne-Hot Encoded Data:")
print(df)
# 4. enerate frequent itemsets
min_support = 0.3
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# 5. Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# 6. Filter rules (confidence & lift)
rules = rules[(rules['confidence'] >= 0.6) & (rules['lift'] > 1)]

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# 7. Visualization - Bar chart of frequent itemsets
plt.figure()
frequent_itemsets['support'].plot(kind='bar')
plt.title("Frequent Itemsets Support")
plt.xlabel("Itemsets")
plt.ylabel("Support")
plt.show()
# 8. Support vs Confidence Plot
plt.figure()
plt.scatter(rules['support'], rules['confidence'])
plt.title("Support vs Confidence")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.show()

# 9. Interpretation (simple print)
print("\nInterpretation:")
for i, row in rules.iterrows():
    print(f"If {set(row['antecedents'])} then {set(row['consequents'])} "
          f"(Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")