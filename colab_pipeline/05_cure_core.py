# Auto-generated from BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

# ---- source cell 32 ----
scaled_embeddings = StandardScaler().fit_transform(user_embeddings)
vis_n = min(len(user_ids), 3000)
vis_idx = np.random.default_rng(SEED).choice(len(user_ids), vis_n, replace=False)
X_vis = scaled_embeddings[vis_idx]
users_vis = np.array(user_ids)[vis_idx]

pca_2d = PCA(n_components=2, random_state=SEED).fit_transform(X_vis)
umap_2d = umap.UMAP(n_components=2, n_neighbors=min(15, max(5, vis_n//20)), min_dist=0.1, random_state=SEED).fit_transform(X_vis)
perplexity = min(30, max(5, (vis_n-1)//3))
tsne_2d = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=SEED).fit_transform(X_vis)

vis_pd = pd.DataFrame({
    'user': users_vis,
    'PCA-1': pca_2d[:,0], 'PCA-2': pca_2d[:,1],
    'UMAP-1': umap_2d[:,0], 'UMAP-2': umap_2d[:,1],
    'tSNE-1': tsne_2d[:,0], 'tSNE-2': tsne_2d[:,1],
})
for method, x, y in [('PCA','PCA-1','PCA-2'), ('UMAP','UMAP-1','UMAP-2'), ('t-SNE','tSNE-1','tSNE-2')]:
    fig = px.scatter(vis_pd, x=x, y=y, hover_data=['user'], title=f'{method} projection of SVD user embeddings')
    fig.update_traces(marker=dict(size=6, opacity=.65))
    fig.show()

# ---- source cell 34 ----
from pyclustering.cluster.cure import cure

CURE_MAX_FULL = 5000
CURE_TUNE_SAMPLE = min(len(user_ids), 1200 if QUALITY_MODE == 'max' else 700)
CURE_KS = list(range(3, min(9, max(4, len(user_ids)//10))))
CURE_REPS = [5, 10] if QUALITY_MODE == 'max' else [5]
CURE_COMPRESSIONS = [0.3, 0.5] if QUALITY_MODE == 'max' else [0.5]

rng = np.random.default_rng(SEED)
tune_idx = rng.choice(len(scaled_embeddings), CURE_TUNE_SAMPLE, replace=False)
X_cure_tune = scaled_embeddings[tune_idx]

def fit_cure_labels(X, k, reps=5, compression=0.5):
    # CCORE first; fallback to Python implementation for compatibility.
    try:
        inst = cure(X.tolist(), k, reps, compression, ccore=True)
        inst.process()
    except Exception:
        inst = cure(X.tolist(), k, reps, compression, ccore=False)
        inst.process()
    clusters = inst.get_clusters()
    labels = np.full(len(X), -1, dtype=int)
    for cid, ids in enumerate(clusters):
        labels[np.array(ids, dtype=int)] = cid
    reps_out = [[np.asarray(p, dtype=float) for p in cluster_reps] for cluster_reps in inst.get_representors()]
    return labels, reps_out

def safe_cluster_metrics(X, labels):
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or np.min(counts) < 2:
        return np.nan, np.nan, np.nan, 0.0
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    balance = float(np.min(counts) / np.max(counts))
    return sil, db, ch, balance

# ---- source cell 35 ----
cure_trials=[]
for k, reps, comp in itertools.product(CURE_KS, CURE_REPS, CURE_COMPRESSIONS):
    t0=time.perf_counter()
    labels, _ = fit_cure_labels(X_cure_tune, k, reps, comp)
    sil, db, ch, balance = safe_cluster_metrics(X_cure_tune, labels)
    cure_trials.append({
        'k':k, 'representatives':reps, 'compression':comp,
        'silhouette':sil, 'davies_bouldin':db, 'calinski_harabasz':ch,
        'balance':balance, 'runtime_sec':time.perf_counter()-t0
    })

cure_tuning_pd=pd.DataFrame(cure_trials).dropna().copy()
if cure_tuning_pd.empty:
    raise RuntimeError('CURE tuning produced no valid multi-cluster solution. Reduce k/representatives or inspect the embedding.')
# Rank-based composite avoids arbitrary metric scales.
cure_tuning_pd['r_sil'] = cure_tuning_pd['silhouette'].rank(pct=True)
cure_tuning_pd['r_db'] = (-cure_tuning_pd['davies_bouldin']).rank(pct=True)
cure_tuning_pd['r_ch'] = cure_tuning_pd['calinski_harabasz'].rank(pct=True)
cure_tuning_pd['cluster_score'] = (
    .40*cure_tuning_pd['r_sil'] + .20*cure_tuning_pd['r_db'] +
    .20*cure_tuning_pd['r_ch'] + .20*cure_tuning_pd['balance']
)
cure_tuning_pd=cure_tuning_pd.sort_values('cluster_score', ascending=False)
display(cure_tuning_pd.head(15))
best_cure=cure_tuning_pd.iloc[0].to_dict()
print('Selected CURE:', best_cure)

# ---- source cell 36 ----
def assign_to_representatives(X, cluster_reps):
    out=np.empty(len(X), dtype=int)
    for i, x in enumerate(X):
        distances=[]
        for reps in cluster_reps:
            R=np.vstack(reps)
            distances.append(float(np.min(np.linalg.norm(R-x, axis=1))))
        out[i]=int(np.argmin(distances))
    return out

def spark_assign_to_representatives(user_ids_local, X, cluster_reps):
    """Distributed nearest-representative assignment for the sampled-CURE branch."""
    reps_serializable=[[p.astype(float).tolist() for p in reps] for reps in cluster_reps]
    bc=sc.broadcast(reps_serializable)
    rows=[(int(u), [float(v) for v in x]) for u,x in zip(user_ids_local,X)]
    sdf=spark.createDataFrame(rows, schema=T.StructType([
        T.StructField('user',T.IntegerType(),False),
        T.StructField('embedding',T.ArrayType(T.DoubleType()),False)
    ]))
    @F.udf(T.IntegerType())
    def nearest_cluster(arr):
        x=np.asarray(arr,dtype=float); ds=[]
        for reps in bc.value:
            R=np.asarray(reps,dtype=float)
            ds.append(float(np.min(np.linalg.norm(R-x,axis=1))))
        return int(np.argmin(ds))
    assigned=sdf.withColumn('cluster',nearest_cluster('embedding')).select('user','cluster')
    mapping={int(r['user']):int(r['cluster']) for r in assigned.collect()}
    bc.unpersist()
    return np.array([mapping[int(u)] for u in user_ids_local],dtype=int)

# Fit final CURE.
if len(scaled_embeddings) <= CURE_MAX_FULL:
    cure_labels, cure_reps = fit_cure_labels(
        scaled_embeddings, int(best_cure['k']), int(best_cure['representatives']), float(best_cure['compression'])
    )
    cure_mode='full CURE'
else:
    sample_idx=np.random.default_rng(SEED).choice(len(scaled_embeddings), CURE_MAX_FULL, replace=False)
    _, cure_reps=fit_cure_labels(
        scaled_embeddings[sample_idx], int(best_cure['k']), int(best_cure['representatives']), float(best_cure['compression'])
    )
    cure_labels=spark_assign_to_representatives(user_ids, scaled_embeddings, cure_reps)
    cure_mode='sampled CURE + Spark representative assignment'

cure_sil, cure_db, cure_ch, cure_balance = safe_cluster_metrics(scaled_embeddings, cure_labels)
print(cure_mode, '| silhouette=', round(cure_sil,4), '| DB=', round(cure_db,4), '| CH=', round(cure_ch,2))
