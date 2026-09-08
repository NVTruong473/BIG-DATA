# Auto-generated from BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

# ---- source cell 60 ----
def spark_top_product_job(df):
    return df.groupBy('itemDescription').count().orderBy(F.desc('count')).limit(100).collect()

def benchmark_scaling(base_spark, scales=(1,10,25,50)):
    base_pd=base_spark.select('itemDescription').toPandas()
    out=[]
    for scale in scales:
        # Spark replication without collecting the large dataset to driver.
        big=(base_spark.select('itemDescription').crossJoin(spark.range(scale).select(F.col('id').alias('_rep'))).drop('_rep'))
        rows=base_spark.count()*scale
        t0=time.perf_counter(); spark_top_product_job(big); ts=time.perf_counter()-t0
        out.append({'scale':scale,'rows':rows,'engine':'Spark','runtime_sec':ts,'throughput_rows_sec':rows/max(ts,1e-9)})

        # Bounded pandas benchmark to avoid turning the demo into an OOM test.
        if rows <= 2_000_000:
            t0=time.perf_counter(); pbig=pd.concat([base_pd]*scale,ignore_index=True)
            _=pbig.groupby('itemDescription').size().nlargest(100)
            tp=time.perf_counter()-t0
            out.append({'scale':scale,'rows':rows,'engine':'pandas','runtime_sec':tp,'throughput_rows_sec':rows/max(tp,1e-9)})
            del pbig
    return pd.DataFrame(out)

SPARK_SCALES=(1,10,25,50) if QUALITY_MODE=='max' else (1,10,25)
scaling_pd=benchmark_scaling(baskets_df,SPARK_SCALES)
display(scaling_pd)

# ---- source cell 61 ----
fig_scaling=px.line(
    scaling_pd,x='rows',y='runtime_sec',color='engine',markers=True,
    hover_data={'scale':True,'throughput_rows_sec':':,.0f'},
    title='Compute scalability — runtime vs rows'
)
fig_scaling.update_xaxes(type='log',title='Rows (log scale)')
fig_scaling.update_yaxes(title='Runtime (seconds)')
fig_scaling.show()

fig_throughput=px.line(
    scaling_pd,x='rows',y='throughput_rows_sec',color='engine',markers=True,
    title='Compute scalability — throughput'
)
fig_throughput.update_xaxes(type='log',title='Rows (log scale)')
fig_throughput.update_yaxes(title='Rows / second')
fig_throughput.show()

# ---- source cell 62 ----
# Cache experiment on the largest scale.
cache_scale=max(SPARK_SCALES)
big_cache=(baskets_df.select('itemDescription').crossJoin(spark.range(cache_scale).select(F.col('id').alias('_rep'))).drop('_rep'))

t0=time.perf_counter(); spark_top_product_job(big_cache); uncached_sec=time.perf_counter()-t0
big_cache=big_cache.cache(); big_cache.count()  # materialize once
t0=time.perf_counter(); spark_top_product_job(big_cache); cached_sec=time.perf_counter()-t0
big_cache.unpersist()
cache_speedup=uncached_sec/max(cached_sec,1e-9)
print(f'Repeated-query cache: uncached={uncached_sec:.3f}s, cached={cached_sec:.3f}s, speedup={cache_speedup:.2f}x')

print('\nPhysical plan of a representative Spark aggregation:')
baskets_df.groupBy('itemDescription').count().orderBy(F.desc('count')).limit(10).explain('formatted')
