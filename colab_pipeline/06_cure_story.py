# Auto-generated from BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

# ---- source cell 37 ----
# Stability validation: fit on repeated 80% samples, then assign all users by CURE representatives.
stability_labelings=[]
stability_sil=[]
for seed in STAT_SEEDS:
    idx=np.random.default_rng(seed).choice(len(scaled_embeddings), max(50, int(.80*len(scaled_embeddings))), replace=False)
    _, reps=fit_cure_labels(
        scaled_embeddings[idx], int(best_cure['k']), int(best_cure['representatives']), float(best_cure['compression'])
    )
    full_labels=assign_to_representatives(scaled_embeddings, reps)
    stability_labelings.append(full_labels)
    stability_sil.append(silhouette_score(scaled_embeddings, full_labels))

ari_values=[]
for a,b in itertools.combinations(stability_labelings,2):
    ari_values.append(adjusted_rand_score(a,b))
cure_stability_ari=float(np.mean(ari_values)) if ari_values else 1.0
print('CURE stability ARI mean=', round(cure_stability_ari,4), '| silhouette mean±std=', round(np.mean(stability_sil),4), '±', round(np.std(stability_sil),4))

# ---- source cell 39 ----
user_behavior_pd = (
    ratings_df.groupBy('user')
    .agg(
        F.count('*').alias('rating_count'),
        F.avg('rating').alias('avg_rating'),
        F.stddev('rating').alias('rating_std'),
        F.countDistinct('item').alias('item_diversity')
    ).toPandas()
)
cluster_users_pd = user_emb_pd[['user']].copy()
cluster_users_pd['cluster'] = cure_labels
cluster_users_pd = cluster_users_pd.merge(user_behavior_pd, on='user', how='left')

cluster_profile = (
    cluster_users_pd.groupby('cluster', as_index=False)
    .agg(users=('user','count'), rating_count=('rating_count','mean'), avg_rating=('avg_rating','mean'),
         rating_std=('rating_std','mean'), item_diversity=('item_diversity','mean'))
)

activity_median=cluster_profile['rating_count'].median()
rating_median=cluster_profile['avg_rating'].median()
def persona(row):
    activity='High-activity' if row.rating_count >= activity_median else 'Selective'
    sentiment='high-rating' if row.avg_rating >= rating_median else 'critical-rating'
    return f'{activity} / {sentiment}'
cluster_profile['persona']=cluster_profile.apply(persona, axis=1)
cluster_users_pd=cluster_users_pd.merge(cluster_profile[['cluster','persona']],on='cluster',how='left')
display(cluster_profile.round(3))

# ---- source cell 40 ----
# Reuse UMAP coordinates for visual cluster story.
vis_cluster = vis_pd.copy()
label_map=dict(zip(user_ids, cure_labels))
profile_map=dict(zip(cluster_profile['cluster'], cluster_profile['persona']))
vis_cluster['cluster']=vis_cluster['user'].map(label_map).astype(str)
vis_cluster['persona']=vis_cluster['user'].map(lambda u: profile_map[label_map[u]])
vis_cluster=vis_cluster.merge(user_behavior_pd,on='user',how='left')

fig_cluster_umap=px.scatter(
    vis_cluster, x='UMAP-1', y='UMAP-2', color='cluster',
    hover_data=['user','persona','rating_count','avg_rating','item_diversity'],
    title='CURE segmentation on SVD embeddings — interactive UMAP'
)
fig_cluster_umap.update_traces(marker=dict(size=7, opacity=.72))
fig_cluster_umap.show()

# ---- source cell 41 ----
# Radar-style normalized cluster profile.
profile_features=['rating_count','avg_rating','rating_std','item_diversity']
prof=cluster_profile.copy()
for c in profile_features:
    lo,hi=prof[c].min(),prof[c].max()
    prof[c+'_n']=(prof[c]-lo)/(hi-lo+1e-9)
fig_cluster_radar=go.Figure()
for _,r in prof.iterrows():
    vals=[r[c+'_n'] for c in profile_features]
    fig_cluster_radar.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=profile_features+[profile_features[0]], fill='toself',
        name=f"Cluster {int(r['cluster'])}: {r['persona']}"
    ))
fig_cluster_radar.update_layout(title='Cluster persona profile (normalized)', polar=dict(radialaxis=dict(visible=True,range=[0,1])))
fig_cluster_radar.show()
