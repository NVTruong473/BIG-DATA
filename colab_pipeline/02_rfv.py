# Auto-generated from BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

# ---- source cell 17 ----
max_date = baskets_df.agg(F.max('date_parsed')).first()[0]

basket_level = (
    baskets_df.filter(F.col('date_parsed').isNotNull())
    .groupBy('Member_number', 'date_parsed')
    .agg(F.count('*').alias('basket_size'))
)

rfv_df = (
    basket_level.groupBy('Member_number')
    .agg(
        F.datediff(F.lit(max_date), F.max('date_parsed')).alias('recency'),
        F.count('*').alias('frequency'),
        F.sum('basket_size').alias('value_proxy'),
        F.avg('basket_size').alias('avg_basket_size')
    )
)

w_r = Window.orderBy(F.col('recency').asc())
w_f = Window.orderBy(F.col('frequency').asc())
w_v = Window.orderBy(F.col('value_proxy').asc())

rfv_df = (
    rfv_df
    .withColumn('R', 6 - F.ntile(5).over(w_r))
    .withColumn('F', F.ntile(5).over(w_f))
    .withColumn('V', F.ntile(5).over(w_v))
    .withColumn('RFV_score', F.col('R') + F.col('F') + F.col('V'))
    .withColumn(
        'segment',
        F.when((F.col('R') >= 4) & (F.col('F') >= 4), 'Champions')
         .when((F.col('F') >= 4) & (F.col('R') >= 2), 'Loyal')
         .when((F.col('R') >= 4) & (F.col('F') <= 2), 'New / Promising')
         .when((F.col('R') <= 2) & (F.col('F') >= 4), 'At Risk')
         .when((F.col('R') <= 2) & (F.col('F') <= 2), 'Hibernating')
         .otherwise('Regular')
    )
)
rfv_pd = rfv_df.toPandas()
display(rfv_pd.groupby('segment').agg(
    customers=('Member_number','count'),
    recency=('recency','mean'),
    frequency=('frequency','mean'),
    value_proxy=('value_proxy','mean')
).round(2).sort_values('customers', ascending=False))

# ---- source cell 18 ----
segment_summary = (
    rfv_pd.groupby('segment', as_index=False)
    .agg(customers=('Member_number','count'), recency=('recency','mean'),
         frequency=('frequency','mean'), value_proxy=('value_proxy','mean'))
)
fig_rfv = px.treemap(
    segment_summary, path=['segment'], values='customers',
    color='frequency',
    hover_data={'recency':':.1f','frequency':':.1f','value_proxy':':.1f'},
    title='RFV customer portfolio — hover to explain each segment'
)
fig_rfv.show()
