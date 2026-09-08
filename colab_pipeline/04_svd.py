# Auto-generated from BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

# ---- source cell 26 ----
user_means_spark = ratings_df.groupBy('user').agg(F.avg('rating').alias('user_mean'))
centered_ratings = (
    ratings_df.join(user_means_spark, 'user')
    .withColumn('centered_rating', F.col('rating') - F.col('user_mean'))
)

item_ids = [r['item'] for r in ratings_df.select('item').distinct().orderBy('item').collect()]
user_item_centered = (
    centered_ratings.groupBy('user').pivot('item', item_ids)
    .agg(F.first('centered_rating')).na.fill(0.0).orderBy('user')
)
user_ids = [int(r['user']) for r in user_item_centered.select('user').collect()]
feature_cols = [str(i) for i in item_ids]

rows_rdd = user_item_centered.select(feature_cols).rdd.map(
    lambda r: OldVectors.dense([float(x or 0.0) for x in r])
).cache()
row_matrix = RowMatrix(rows_rdd)

max_possible = max(2, min(len(user_ids)-1, len(item_ids)-1, 64))
SVD_CANDIDATES = [k for k in [4,8,16,32,64] if k <= max_possible]
max_k = max(SVD_CANDIDATES)
print('SVD candidates:', SVD_CANDIDATES, '| max_k:', max_k)

# ---- source cell 27 ----
t0 = time.perf_counter()
svd_full = row_matrix.computeSVD(max_k, computeU=True)
svd_runtime = time.perf_counter() - t0

singular_values = np.array(svd_full.s, dtype=float)
total_energy = rows_rdd.map(lambda v: float(np.dot(v.toArray(), v.toArray()))).sum()
cum_energy = np.cumsum(singular_values**2) / max(total_energy, 1e-12)

svd_curve = pd.DataFrame({
    'component': np.arange(1, len(singular_values)+1),
    'singular_value': singular_values,
    'captured_energy': cum_energy
})

selected_svd_k = None
for k in SVD_CANDIDATES:
    if cum_energy[k-1] >= 0.90:
        selected_svd_k = k
        break
if selected_svd_k is None:
    selected_svd_k = max_k

print(f'Selected latent dimension = {selected_svd_k}; captured energy={cum_energy[selected_svd_k-1]:.2%}; runtime={svd_runtime:.2f}s')

# ---- source cell 28 ----
fig_svd = make_subplots(specs=[[{'secondary_y': True}]])
fig_svd.add_trace(go.Bar(x=svd_curve['component'], y=svd_curve['singular_value'], name='Singular value'), secondary_y=False)
fig_svd.add_trace(go.Scatter(x=svd_curve['component'], y=svd_curve['captured_energy'], name='Cumulative captured energy', mode='lines+markers'), secondary_y=True)
fig_svd.add_hline(y=0.90, line_dash='dash', annotation_text='90% target', secondary_y=True)
fig_svd.add_vline(x=selected_svd_k, line_dash='dot', annotation_text=f'Selected k={selected_svd_k}')
fig_svd.update_layout(title='SVD latent-dimension evidence', hovermode='x unified')
fig_svd.update_yaxes(title_text='Singular value', secondary_y=False)
fig_svd.update_yaxes(title_text='Captured energy', tickformat='.0%', secondary_y=True)
fig_svd.show()

# ---- source cell 29 ----
# Extract UΣ and VΣ embeddings.
U_np = np.vstack([v.toArray() for v in svd_full.U.rows.collect()])[:, :selected_svd_k]
S_np = singular_values[:selected_svd_k]
V_np = svd_full.V.toArray()[:, :selected_svd_k]

user_embeddings = U_np * S_np
item_embeddings = V_np * S_np

user_emb_pd = pd.DataFrame(user_embeddings, columns=[f'z{i+1}' for i in range(selected_svd_k)])
user_emb_pd.insert(0, 'user', user_ids)
item_emb_pd = pd.DataFrame(item_embeddings, columns=[f'z{i+1}' for i in range(selected_svd_k)])
item_emb_pd.insert(0, 'item', item_ids)

user_emb_pd.to_csv(OUTPUT_DIR/'user_svd_embeddings.csv', index=False)
item_emb_pd.to_csv(OUTPUT_DIR/'item_svd_embeddings.csv', index=False)
display(user_emb_pd.head())

# ---- source cell 30 ----
# Preserve the original Task 1 deliverables: assign each user/item to its dominant latent concept.
# Dominant concept = latent dimension with the largest absolute embedding magnitude.
user_concepts_pd = pd.DataFrame({
    'user': user_ids,
    'concept': np.argmax(np.abs(user_embeddings), axis=1).astype(int) + 1,
    'concept_strength': np.max(np.abs(user_embeddings), axis=1)
})
item_concepts_pd = pd.DataFrame({
    'item': item_ids,
    'concept': np.argmax(np.abs(item_embeddings), axis=1).astype(int) + 1,
    'concept_strength': np.max(np.abs(item_embeddings), axis=1)
})
user_concepts_pd.to_csv(OUTPUT_DIR/'concept_user.csv', index=False)
item_concepts_pd.to_csv(OUTPUT_DIR/'concept_item.csv', index=False)
display(user_concepts_pd.head())
display(item_concepts_pd.head())
