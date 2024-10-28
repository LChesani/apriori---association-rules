import os, json, pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
rawData = None
file_path = 'database/padaria_trab.json'
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f: #o encoding ta tratando os caracteres especiais
        rawData = json.load(f)
else:
    print(f"Error: The file '{file_path}' was not found.")

#Data Preprocessing
#na linha 737 tinha uma virgula faltando, foi adicionada manualmente no arquivo
data = []
for product in rawData: #como compra era uma coluna com valor nao significativo, foi removida, posso restaurar ela caso necessario setando um id
    data.append(product['produtos']) #rawData é uma lista de dicionarios, transformamos em uma lista de listas de compras

data = [[item.split()[0] for item in sublista] for sublista in data]

#Data Transformation
#Transformando em formato tabular
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df = pd.DataFrame(te_ary, columns=te.columns_)

#Data Mining
#apriori e regras
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(frequent_itemsets)
print(regras)
