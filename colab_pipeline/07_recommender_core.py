# Auto-generated from BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

# ---- source cell 43 ----
ratings_pd = ratings_df.toPandas().astype({'user':int,'item':int,'rating':float})
RATING_LOW=float(ratings_pd['rating'].min()); RATING_HIGH=float(ratings_pd['rating'].max())
if RATING_HIGH < RELEVANCE_THRESHOLD:
    RELEVANCE_THRESHOLD=float(ratings_pd['rating'].quantile(.75))
print('Relevance threshold =', RELEVANCE_THRESHOLD)

def per_user_split(df, seed=42, test_frac=.2, min_ratings=5):
    rng=np.random.default_rng(seed)
    train_idx=[]; test_idx=[]
    for _,g in df.groupby('user'):
        ids=g.index.to_numpy()
        if len(ids) < min_ratings:
            train_idx.extend(ids.tolist()); continue
        n_test=max(1,int(round(len(ids)*test_frac)))
        n_test=min(n_test, len(ids)-2)
        chosen=set(rng.choice(ids, n_test, replace=False).tolist())
        test_idx.extend(chosen)
        train_idx.extend([i for i in ids if i not in chosen])
    train=df.loc[train_idx].copy(); test=df.loc[test_idx].copy()
    # prevent item cold start in test
    train_items=set(train['item'])
    cold=test[~test['item'].isin(train_items)]
    if len(cold):
        train=pd.concat([train,cold],ignore_index=False)
        test=test.drop(index=cold.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)

def rating_metrics(y_true,y_pred):
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true,y_pred))),
        'MAE': float(mean_absolute_error(y_true,y_pred))
    }

def ranking_metrics(pred_dict, test_df, k=10, threshold=4.0):
    relevant=(test_df[test_df['rating']>=threshold].groupby('user')['item'].apply(set).to_dict())
    users=[u for u,s in relevant.items() if len(s)>0 and u in pred_dict]
    if not users: return {'Precision@K':0.0,'Recall@K':0.0,'eval_users':0}
    ps=[]; rs=[]
    for u in users:
        pred=list(pred_dict[u])[:k]; gt=relevant[u]
        hits=len(set(pred)&gt)
        ps.append(hits/k)
        rs.append(hits/len(gt))
    return {'Precision@K':float(np.mean(ps)),'Recall@K':float(np.mean(rs)),'eval_users':len(users)}

# ---- source cell 44 ----
# ---------- UserCF ----------
def build_usercf_state(train):
    users=sorted(train.user.unique()); items=sorted(train.item.unique())
    pivot=train.pivot_table(index='user',columns='item',values='rating',aggfunc='mean').reindex(index=users,columns=items)
    A=pivot.to_numpy(dtype=float); mask=~np.isnan(A)
    global_mean=float(train.rating.mean())
    user_mean=np.nanmean(A,axis=1)
    user_mean=np.where(np.isnan(user_mean),global_mean,user_mean)
    centered=np.where(mask,A-user_mean[:,None],0.0)
    norms=np.linalg.norm(centered,axis=1)
    sim=centered@centered.T/(np.outer(norms,norms)+1e-12)
    sim=np.where(sim>0,sim,0.0); np.fill_diagonal(sim,0.0)
    return {'users':users,'items':items,'u2i':{u:i for i,u in enumerate(users)},'i2i':{it:i for i,it in enumerate(items)},
            'A':A,'mask':mask,'user_mean':user_mean,'centered':centered,'sim':sim,'global_mean':global_mean,
            'item_mean':train.groupby('item').rating.mean().to_dict()}

def usercf_similarity_topn(sim,n_neighbors):
    if n_neighbors>=sim.shape[1]: return sim.copy()
    out=np.zeros_like(sim)
    idx=np.argpartition(sim,-n_neighbors,axis=1)[:,-n_neighbors:]
    rows=np.arange(sim.shape[0])[:,None]
    out[rows,idx]=sim[rows,idx]
    return out

def usercf_predict(state, pairs, n_neighbors=40):
    sim=usercf_similarity_topn(state['sim'],n_neighbors)
    pred=[]
    for r in pairs.itertuples():
        if r.user not in state['u2i']:
            p=state['item_mean'].get(r.item,state['global_mean']); pred.append(p); continue
        ui=state['u2i'][r.user]
        if r.item not in state['i2i']:
            pred.append(state['user_mean'][ui]); continue
        ii=state['i2i'][r.item]
        raters=state['mask'][:,ii]
        w=sim[ui]*raters
        den=np.sum(np.abs(w))
        p=state['user_mean'][ui] if den<1e-12 else state['user_mean'][ui]+np.sum(w*state['centered'][:,ii])/den
        pred.append(float(np.clip(p,RATING_LOW,RATING_HIGH)))
    return np.asarray(pred)

def usercf_recommend(state, users, k=10, n_neighbors=40):
    sim=usercf_similarity_topn(state['sim'],n_neighbors)
    num=sim@state['centered']; den=np.abs(sim)@state['mask'].astype(float)
    score=state['user_mean'][:,None]+np.divide(num,den,out=np.zeros_like(num),where=den>1e-12)
    out={}
    items=np.array(state['items'])
    for u in users:
        if u not in state['u2i']: continue
        ui=state['u2i'][u]; s=score[ui].copy(); s[state['mask'][ui]]=-np.inf
        take=min(k,len(s)); idx=np.argpartition(s,-take)[-take:]; idx=idx[np.argsort(s[idx])[::-1]]
        out[u]=items[idx].astype(int).tolist()
    return out

# ---- source cell 45 ----
# ---------- SVD recommender ----------
def build_svd_recommender(train, n_components=16):
    users=sorted(train.user.unique()); items=sorted(train.item.unique())
    pivot=train.pivot_table(index='user',columns='item',values='rating',aggfunc='mean').reindex(index=users,columns=items)
    A=pivot.to_numpy(dtype=float); mask=~np.isnan(A); global_mean=float(train.rating.mean())
    user_mean=np.nanmean(A,axis=1); user_mean=np.where(np.isnan(user_mean),global_mean,user_mean)
    centered=np.where(mask,A-user_mean[:,None],0.0)
    k=max(2,min(n_components,min(centered.shape)-1))
    model=TruncatedSVD(n_components=k,random_state=SEED)
    latent=model.fit_transform(centered); recon=latent@model.components_ + user_mean[:,None]
    return {'users':users,'items':items,'u2i':{u:i for i,u in enumerate(users)},'i2i':{it:i for i,it in enumerate(items)},
            'mask':mask,'pred':np.clip(recon,RATING_LOW,RATING_HIGH),'global_mean':global_mean,
            'user_mean':user_mean,'item_mean':train.groupby('item').rating.mean().to_dict(),'k':k}

def svd_predict(state,pairs):
    out=[]
    for r in pairs.itertuples():
        if r.user in state['u2i'] and r.item in state['i2i']:
            out.append(state['pred'][state['u2i'][r.user],state['i2i'][r.item]])
        elif r.item in state['item_mean']: out.append(state['item_mean'][r.item])
        else: out.append(state['global_mean'])
    return np.asarray(out)

def svd_recommend(state,users,k=10):
    items=np.array(state['items']); out={}
    for u in users:
        if u not in state['u2i']: continue
        ui=state['u2i'][u]; s=state['pred'][ui].copy(); s[state['mask'][ui]]=-np.inf
        take=min(k,len(s)); idx=np.argpartition(s,-take)[-take:]; idx=idx[np.argsort(s[idx])[::-1]]
        out[u]=items[idx].astype(int).tolist()
    return out
