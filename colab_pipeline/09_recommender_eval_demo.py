# Auto-generated from BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

# ---- source cell 52 ----
def popularity_recommend(train, users, k=10):
    pop=(train.assign(relevant=train.rating>=RELEVANCE_THRESHOLD)
         .groupby('item').agg(rel=('relevant','sum'),count=('rating','count'),avg=('rating','mean'))
         .sort_values(['rel','avg','count'],ascending=False).index.astype(int).tolist())
    seen=train.groupby('user').item.apply(set).to_dict(); out={}
    for u in users: out[u]=[it for it in pop if it not in seen.get(u,set())][:k]
    return out

def evaluate_models_once(seed):
    train,test=per_user_split(ratings_pd,seed,TEST_FRAC,MIN_USER_RATINGS_FOR_TEST)
    eval_users=test[test.rating>=RELEVANCE_THRESHOLD].user.unique().tolist()
    rating_rows=[]; ranking_rows=[]
    global_mean=float(train.rating.mean()); user_mean=train.groupby('user').rating.mean().to_dict(); item_mean=train.groupby('item').rating.mean().to_dict()
    baseline_preds={
        'GlobalMean':np.full(len(test),global_mean),
        'UserMean':np.array([user_mean.get(u,global_mean) for u in test.user]),
        'ItemMean':np.array([item_mean.get(i,global_mean) for i in test.item]),
    }
    for name,p in baseline_preds.items(): rating_rows.append({'seed':seed,'model':name,**rating_metrics(test.rating,p)})

    # UserCF
    ustate=build_usercf_state(train); up=usercf_predict(ustate,test,best_ucf)
    rating_rows.append({'seed':seed,'model':'UserCF',**rating_metrics(test.rating,up)})
    urec=usercf_recommend(ustate,eval_users,TOP_K,best_ucf)
    ranking_rows.append({'seed':seed,'model':'UserCF',**ranking_metrics(urec,test,TOP_K,RELEVANCE_THRESHOLD)})

    # SVD reconstruction
    sstate=build_svd_recommender(train,best_svd_rec_k); sp=svd_predict(sstate,test)
    rating_rows.append({'seed':seed,'model':'SVD-Reconstruction',**rating_metrics(test.rating,sp)})
    srec=svd_recommend(sstate,eval_users,TOP_K)
    ranking_rows.append({'seed':seed,'model':'SVD-Reconstruction',**ranking_metrics(srec,test,TOP_K,RELEVANCE_THRESHOLD)})

    # Popularity ranking baseline
    prec=popularity_recommend(train,eval_users,TOP_K)
    ranking_rows.append({'seed':seed,'model':'Popularity',**ranking_metrics(prec,test,TOP_K,RELEVANCE_THRESHOLD)})

    # ALS
    amodel=fit_als(train,best_als,seed); am=als_rating_eval(amodel,test)
    rating_rows.append({'seed':seed,'model':'ALS',**am})
    arec=als_recommend_dict(amodel,train,eval_users,TOP_K)
    ranking_rows.append({'seed':seed,'model':'ALS',**ranking_metrics(arec,test,TOP_K,RELEVANCE_THRESHOLD)})
    return rating_rows,ranking_rows,amodel,train,test,arec

all_rating=[]; all_ranking=[]; first_als_model=None; first_train=None; first_test=None; first_als_recs=None
for seed in STAT_SEEDS:
    rr,rk,amodel,tr,te,recs=evaluate_models_once(seed)
    all_rating.extend(rr); all_ranking.extend(rk)
    if first_als_model is None:
        first_als_model,first_train,first_test,first_als_recs=amodel,tr,te,recs

rating_runs=pd.DataFrame(all_rating); ranking_runs=pd.DataFrame(all_ranking)
rating_stats=rating_runs.groupby('model').agg(RMSE_mean=('RMSE','mean'),RMSE_std=('RMSE','std'),MAE_mean=('MAE','mean'),MAE_std=('MAE','std')).reset_index()
ranking_stats=ranking_runs.groupby('model').agg(Precision_mean=('Precision@K','mean'),Precision_std=('Precision@K','std'),Recall_mean=('Recall@K','mean'),Recall_std=('Recall@K','std'),eval_users=('eval_users','mean')).reset_index()

display(rating_stats.sort_values('RMSE_mean'))
display(ranking_stats.sort_values('Recall_mean',ascending=False))

# ---- source cell 54 ----
first_labels=(first_test[first_test.rating>=RELEVANCE_THRESHOLD].groupby('user').item.apply(list).to_dict())
rank_rows=[]
for u,labels in first_labels.items():
    if u in first_als_recs and labels:
        rank_rows.append(([float(x) for x in first_als_recs[u]],[float(x) for x in labels]))
if rank_rows:
    rank_eval_df=spark.createDataFrame(rank_rows,['prediction','label'])
    spark_precision=RankingEvaluator(predictionCol='prediction',labelCol='label',metricName='precisionAtK',k=TOP_K).evaluate(rank_eval_df)
    spark_recall=RankingEvaluator(predictionCol='prediction',labelCol='label',metricName='recallAtK',k=TOP_K).evaluate(rank_eval_df)
    manual_first=ranking_runs[(ranking_runs.seed==STAT_SEEDS[0])&(ranking_runs.model=='ALS')].iloc[0]
    print(f'Spark Precision@{TOP_K}={spark_precision:.6f} | manual={manual_first["Precision@K"]:.6f}')
    print(f'Spark Recall@{TOP_K}={spark_recall:.6f} | manual={manual_first["Recall@K"]:.6f}')
else:
    print('No users with held-out relevant items for RankingEvaluator cross-check.')

# ---- source cell 55 ----
# Automatic champion score for models that have both rating and ranking metrics.
model_score=rating_stats.merge(ranking_stats,on='model',how='inner')
for col in ['RMSE_mean','MAE_mean','Precision_mean','Recall_mean']:
    lo,hi=model_score[col].min(),model_score[col].max()
    model_score[col+'_n']=(model_score[col]-lo)/(hi-lo+1e-9)
model_score['ChampionScore']=(
    .25*(1-model_score['RMSE_mean_n']) + .25*(1-model_score['MAE_mean_n']) +
    .25*model_score['Precision_mean_n'] + .25*model_score['Recall_mean_n']
)
model_score=model_score.sort_values('ChampionScore',ascending=False)
champion=model_score.iloc[0]
display(model_score[['model','RMSE_mean','MAE_mean','Precision_mean','Recall_mean','ChampionScore']])
print('MODEL CHAMPION:',champion['model'])

# ---- source cell 57 ----
DEMO_USER = int(sorted(first_train.user.unique())[0])
champion_name = str(champion['model'])

def champion_recommendation_demo(user_id, k=10):
    seen=set(first_train[first_train.user==user_id].item.astype(int))
    if champion_name == 'ALS':
        users_sdf=spark.createDataFrame([(int(user_id),)],['user'])
        raw=first_als_model.recommendForUserSubset(users_sdf,min(len(set(first_train.item)),max(100,k*10)))
        rows=(raw.select('user',F.explode('recommendations').alias('rec'))
              .select(F.col('rec.item').alias('item'),F.col('rec.rating').alias('predicted_rating')).collect())
        data=[(int(r['item']),float(r['predicted_rating'])) for r in rows if int(r['item']) not in seen][:k]
        return pd.DataFrame(data,columns=['item','predicted_rating'])
    if champion_name == 'UserCF':
        st=build_usercf_state(first_train); rec=usercf_recommend(st,[user_id],k,best_ucf).get(user_id,[])
        pairs=pd.DataFrame({'user':[user_id]*len(rec),'item':rec})
        pairs['rating']=0.0
        pairs['predicted_rating']=usercf_predict(st,pairs,best_ucf) if len(pairs) else []
        return pairs[['item','predicted_rating']]
    if champion_name == 'SVD-Reconstruction':
        st=build_svd_recommender(first_train,best_svd_rec_k); rec=svd_recommend(st,[user_id],k).get(user_id,[])
        if not rec: return pd.DataFrame(columns=['item','predicted_rating'])
        ui=st['u2i'][user_id]
        return pd.DataFrame({'item':rec,'predicted_rating':[float(st['pred'][ui,st['i2i'][it]]) for it in rec]})
    return pd.DataFrame(columns=['item','predicted_rating'])

customer_demo_pd=champion_recommendation_demo(DEMO_USER,TOP_K)
customer_demo_pd.insert(0,'rank',np.arange(1,len(customer_demo_pd)+1))
print(f'Demo user={DEMO_USER} | champion={champion_name} | already-rated items={len(first_train[first_train.user==DEMO_USER])}')
display(customer_demo_pd)

if len(customer_demo_pd):
    fig_customer_demo=px.bar(
        customer_demo_pd.sort_values('rank',ascending=False),
        x='predicted_rating',y=customer_demo_pd.sort_values('rank',ascending=False)['item'].astype(str),orientation='h',
        hover_data={'rank':True,'predicted_rating':':.3f'},
        title=f'Customer-facing Top-{TOP_K} recommendations — user {DEMO_USER}'
    )
    fig_customer_demo.update_yaxes(title='Item')
    fig_customer_demo.show()

# ---- source cell 58 ----
fig_model_quality=make_subplots(rows=1,cols=2,subplot_titles=('Rating accuracy — lower is better','Top-N quality — higher is better'))
rs=rating_stats.sort_values('RMSE_mean')
fig_model_quality.add_trace(go.Bar(x=rs.model,y=rs.RMSE_mean,error_y=dict(type='data',array=rs.RMSE_std.fillna(0)),name='RMSE'),row=1,col=1)
fig_model_quality.add_trace(go.Bar(x=rs.model,y=rs.MAE_mean,error_y=dict(type='data',array=rs.MAE_std.fillna(0)),name='MAE'),row=1,col=1)
ks=ranking_stats.sort_values('Recall_mean',ascending=False)
fig_model_quality.add_trace(go.Bar(x=ks.model,y=ks.Precision_mean,error_y=dict(type='data',array=ks.Precision_std.fillna(0)),name='Precision@10'),row=1,col=2)
fig_model_quality.add_trace(go.Bar(x=ks.model,y=ks.Recall_mean,error_y=dict(type='data',array=ks.Recall_std.fillna(0)),name='Recall@10'),row=1,col=2)
fig_model_quality.update_layout(title='Recommendation benchmark — mean ± std across seeds',barmode='group',height=520)
fig_model_quality.show()
