# Auto-generated from BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

# ---- source cell 13 ----
def read_csv_rdd(file_path):
    rdd = sc.textFile(file_path)
    header = rdd.first()
    return rdd.filter(lambda line: line != header).map(lambda line: line.split(','))

def f1_rdd(file_path, top_n=100):
    return (
        read_csv_rdd(file_path)
        .map(lambda x: (x[2], 1))
        .reduceByKey(lambda a, b: a + b)
        .takeOrdered(top_n, key=lambda x: -x[1])
    )

def f2_rdd(file_path, top_n=100):
    # Basket = Member_number + Date.
    return (
        read_csv_rdd(file_path)
        .map(lambda x: ((x[0], x[1]), 1))
        .keys().distinct()
        .map(lambda x: (x[0], 1))
        .reduceByKey(lambda a, b: a + b)
        .takeOrdered(top_n, key=lambda x: -x[1])
    )

def f3_rdd(file_path):
    return (
        read_csv_rdd(file_path)
        .map(lambda x: ((int(x[3]), int(x[4]), x[2]), 1))
        .reduceByKey(lambda a, b: a + b)
    )

def f4_rdd(file_path):
    return (
        read_csv_rdd(file_path)
        .map(lambda x: ((x[0], x[1]), (int(x[3]), int(x[4]))))
        .distinct()
        .map(lambda kv: (kv[1], 1))
        .reduceByKey(lambda a, b: a + b)
        .sortByKey()
        .collect()
    )

# ---- source cell 14 ----
# DataFrame equivalents
f1_df = (
    baskets_df.groupBy('itemDescription').count()
    .orderBy(F.desc('count')).limit(100)
)

f2_df = (
    baskets_df.select('Member_number', 'Date').distinct()
    .groupBy('Member_number').count()
    .orderBy(F.desc('count')).limit(100)
)

f4_df = (
    baskets_df.select('Member_number', 'Date', 'year', 'month').distinct()
    .groupBy('year', 'month').count()
    .orderBy('year', 'month')
)

print('Top products — RDD')
display(pd.DataFrame(f1_rdd(BASKETS_PATH, 20), columns=['product','count']))
print('Top customers by distinct baskets — RDD')
display(pd.DataFrame(f2_rdd(BASKETS_PATH, 20), columns=['member','basket_count']))
print('Monthly distinct baskets — RDD')
display(pd.DataFrame(f4_rdd(BASKETS_PATH), columns=['year_month','basket_count']))

# ---- source cell 15 ----
top_products_pd = f1_df.limit(20).toPandas().sort_values('count')
fig_top_products = px.bar(
    top_products_pd, x='count', y='itemDescription', orientation='h',
    title='Top 20 purchased products',
    labels={'count':'Purchases', 'itemDescription':'Product'}
)
fig_top_products.update_traces(hovertemplate='%{y}<br>Purchases=%{x}<extra></extra>')
fig_top_products.show()
