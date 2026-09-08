# Auto-generated from BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

# ---- source cell 46 ----
# ---------- Spark ALS ----------
def to_spark_ratings(pdf):
    return spark.createDataFrame(pdf[['user','item','rating']].astype({'user':int,'item':int,'rating':float}))

def fit_als(train_pdf, params, seed=42):
    als=ALS(
        userCol='user', itemCol='item', ratingCol='rating',
        rank=int(params['rank']), regParam=float(params['regParam']), maxIter=int(params['maxIter']),
        coldStartStrategy='drop', seed=int(seed), nonnegative=False
    )
    return als.fit(to_spark_ratings(train_pdf))

def als_rating_eval(model,test_pdf):
    pred=model.transform(to_spark_ratings(test_pdf)).select('rating','prediction').dropna().toPandas()
    if len(pred)==0: return {'RMSE':np.nan,'MAE':np.nan}
    return rating_metrics(pred.rating,pred.prediction)

def als_recommend_dict(model,train_pdf,eval_users,k=10):
    users_sdf=spark.createDataFrame([(int(u),) for u in sorted(set(eval_users))],['user'])
    request_n=min(len(set(train_pdf['item'])),max(100,k*10))
    rec=model.recommendForUserSubset(users_sdf,request_n)
    exploded=(
        rec.select('user',F.posexplode('recommendations').alias('pos','rec'))
        .select('user',F.col('rec.item').alias('item'),F.col('rec.rating').alias('score'))
    )
    seen=to_spark_ratings(train_pdf).select('user','item').distinct()
    unseen=exploded.join(seen,['user','item'],'left_anti')
    w=Window.partitionBy('user').orderBy(F.desc('score'))
    top=(unseen.withColumn('rn',F.row_number().over(w)).filter(F.col('rn')<=k)
         .groupBy('user').agg(F.sort_array(F.collect_list(F.struct('rn','item'))).alias('arr')))
    rows=top.collect()
    return {int(r['user']):[int(x['item']) for x in r['arr']] for r in rows}

# ---- source cell 48 ----
master_train, master_test = per_user_split(ratings_pd, SEED, TEST_FRAC, MIN_USER_RATINGS_FOR_TEST)
inner_train, inner_val = per_user_split(master_train, SEED+100, .15, MIN_USER_RATINGS_FOR_TEST)
val_users=inner_val[inner_val.rating>=RELEVANCE_THRESHOLD].user.unique().tolist()

# UserCF tuning
ucf_state=build_usercf_state(inner_train)
ucf_trials=[]
for nn in ([10,20,40,80] if QUALITY_MODE=='max' else [20,40]):
    p=usercf_predict(ucf_state,inner_val,nn); m=rating_metrics(inner_val.rating,p)
    rec=usercf_recommend(ucf_state,val_users,TOP_K,nn); r=ranking_metrics(rec,inner_val,TOP_K,RELEVANCE_THRESHOLD)
    ucf_trials.append({'neighbors':nn,**m,**r})
ucf_tuning=pd.DataFrame(ucf_trials)
# normalized balanced score
for col in ['RMSE','Recall@K']:
    lo,hi=ucf_tuning[col].min(),ucf_tuning[col].max(); ucf_tuning[col+'_n']=(ucf_tuning[col]-lo)/(hi-lo+1e-9)
ucf_tuning['score']=.6*(1-ucf_tuning['RMSE_n'])+.4*ucf_tuning['Recall@K_n']
best_ucf=int(ucf_tuning.sort_values('score',ascending=False).iloc[0]['neighbors'])
display(ucf_tuning)
print('Best UserCF neighbors=',best_ucf)

# ---- source cell 49 ----
# SVD recommender tuning
svd_rec_candidates=sorted(set([k for k in [4,8,16,32,selected_svd_k] if k < min(inner_train.user.nunique(),inner_train.item.nunique())]))
svd_trials=[]
for k in svd_rec_candidates:
    st=build_svd_recommender(inner_train,k); p=svd_predict(st,inner_val); m=rating_metrics(inner_val.rating,p)
    rec=svd_recommend(st,val_users,TOP_K); r=ranking_metrics(rec,inner_val,TOP_K,RELEVANCE_THRESHOLD)
    svd_trials.append({'components':k,**m,**r})
svd_tuning=pd.DataFrame(svd_trials)
for col in ['RMSE','Recall@K']:
    lo,hi=svd_tuning[col].min(),svd_tuning[col].max(); svd_tuning[col+'_n']=(svd_tuning[col]-lo)/(hi-lo+1e-9)
svd_tuning['score']=.6*(1-svd_tuning['RMSE_n'])+.4*svd_tuning['Recall@K_n']
best_svd_rec_k=int(svd_tuning.sort_values('score',ascending=False).iloc[0]['components'])
display(svd_tuning)
print('Best SVD recommender components=',best_svd_rec_k)

# ---- source cell 50 ----
# ALS tuning — broad enough for demo, bounded to stay below the requested runtime budget.
ALS_RANKS=[8,16,32] if QUALITY_MODE=='max' else [8,16]
ALS_REGS=[0.03,0.10,0.30] if QUALITY_MODE=='max' else [0.05,0.15]
ALS_ITERS=[10,20] if QUALITY_MODE=='max' else [10]

als_trials=[]
for rank,reg,iters in itertools.product(ALS_RANKS,ALS_REGS,ALS_ITERS):
    t0=time.perf_counter(); params={'rank':rank,'regParam':reg,'maxIter':iters}
    model=fit_als(inner_train,params,SEED)
    m=als_rating_eval(model,inner_val)
    als_trials.append({**params,**m,'runtime_sec':time.perf_counter()-t0})
als_tuning=pd.DataFrame(als_trials).sort_values('RMSE')

# Rank quality only on the best RMSE candidates.
top_candidates=als_tuning.head(min(4,len(als_tuning))).copy()
rank_rows=[]
for _,r in top_candidates.iterrows():
    params={'rank':int(r['rank']),'regParam':float(r['regParam']),'maxIter':int(r['maxIter'])}
    model=fit_als(inner_train,params,SEED)
    rec=als_recommend_dict(model,inner_train,val_users,TOP_K)
    rank_rows.append({**params,**ranking_metrics(rec,inner_val,TOP_K,RELEVANCE_THRESHOLD)})
als_rank_tuning=pd.DataFrame(rank_rows)
als_joint=top_candidates.merge(als_rank_tuning,on=['rank','regParam','maxIter'])

lo,hi=als_joint.RMSE.min(),als_joint.RMSE.max(); als_joint['rmse_n']=(als_joint.RMSE-lo)/(hi-lo+1e-9)
lo,hi=als_joint['Recall@K'].min(),als_joint['Recall@K'].max(); als_joint['recall_n']=(als_joint['Recall@K']-lo)/(hi-lo+1e-9)
als_joint['selection_score']=.60*(1-als_joint['rmse_n'])+.40*als_joint['recall_n']
best_als_row=als_joint.sort_values('selection_score',ascending=False).iloc[0]
best_als={'rank':int(best_als_row['rank']),'regParam':float(best_als_row['regParam']),'maxIter':int(best_als_row['maxIter'])}
display(als_joint.sort_values('selection_score',ascending=False))
print('Best ALS params=',best_als)
