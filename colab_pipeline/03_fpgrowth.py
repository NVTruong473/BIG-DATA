# Auto-generated from BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

# ---- source cell 20 ----
fp_baskets = (
    baskets_df.groupBy('Member_number', 'Date')
    .agg(F.collect_set('itemDescription').alias('Items'))
    .filter(F.size('Items') >= 2)
    .cache()
)
print('Baskets used by FP-Growth:', fp_baskets.count())

# ---- source cell 21 ----
FP_SUPPORTS = [0.003, 0.005, 0.01, 0.02] if QUALITY_MODE == 'max' else [0.005, 0.01, 0.02]
FP_CONFIDENCES = [0.05, 0.10, 0.20] if QUALITY_MODE == 'max' else [0.05, 0.10]

fp_trials = []
for min_support, min_conf in itertools.product(FP_SUPPORTS, FP_CONFIDENCES):
    t0 = time.perf_counter()
    model = FPGrowth(itemsCol='Items', minSupport=min_support, minConfidence=min_conf).fit(fp_baskets)
    rules = model.associationRules
    s = rules.agg(
        F.count('*').alias('n_rules'),
        F.avg('confidence').alias('avg_confidence'),
        F.avg('lift').alias('avg_lift'),
        F.max('lift').alias('max_lift'),
        F.avg('support').alias('avg_support')
    ).first()
    n_rules = int(s['n_rules'] or 0)
    avg_conf = float(s['avg_confidence'] or 0)
    avg_lift = float(s['avg_lift'] or 0)
    avg_support = float(s['avg_support'] or 0)
    # Prefer useful but reviewable rule volume. Hard penalty for pathological output sizes.
    volume_factor = min(1.0, n_rules / 30.0) * min(1.0, 800.0 / max(n_rules, 1))
    actionability = volume_factor * avg_conf * max(avg_lift, 0) * math.sqrt(max(avg_support, 1e-9))
    fp_trials.append({
        'minSupport': min_support, 'minConfidence': min_conf, 'rules': n_rules,
        'avg_confidence': avg_conf, 'avg_lift': avg_lift, 'max_lift': float(s['max_lift'] or 0),
        'avg_support': avg_support, 'actionability': actionability,
        'runtime_sec': time.perf_counter() - t0
    })

fp_tuning_pd = pd.DataFrame(fp_trials).sort_values('actionability', ascending=False)
display(fp_tuning_pd)
best_fp = fp_tuning_pd.iloc[0].to_dict()
print('Selected FP-Growth params:', best_fp)

# ---- source cell 22 ----
fp_model = FPGrowth(
    itemsCol='Items',
    minSupport=float(best_fp['minSupport']),
    minConfidence=float(best_fp['minConfidence'])
).fit(fp_baskets)

fp_rules = (
    fp_model.associationRules
    .withColumn('action_score', F.col('confidence') * F.col('lift') * F.sqrt(F.col('support')))
    .orderBy(F.desc('action_score'))
)
fp_rules_pd = fp_rules.limit(200).toPandas()
fp_rules_pd['antecedent_text'] = fp_rules_pd['antecedent'].apply(lambda x: ' + '.join(x))
fp_rules_pd['consequent_text'] = fp_rules_pd['consequent'].apply(lambda x: ' + '.join(x))
display(fp_rules_pd[['antecedent_text','consequent_text','confidence','lift','support','action_score']].head(30))

# ---- source cell 23 ----
fig_rules = px.scatter(
    fp_rules_pd,
    x='confidence', y='lift', size='support', color='action_score',
    hover_name='antecedent_text', hover_data={'consequent_text':True,'support':':.4f','action_score':':.4f'},
    title='Association Rules — Confidence × Lift × Support'
)
fig_rules.add_hline(y=1.0, line_dash='dash', annotation_text='Lift = 1 (no positive association)')
fig_rules.show()

# ---- source cell 24 ----
# Interactive product-rule network for top actionable rules.
def build_rule_network(rules_pd, n=25):
    d = rules_pd.head(n).copy()
    G = nx.DiGraph()
    for _, r in d.iterrows():
        a = r['antecedent_text']; c = r['consequent_text']
        G.add_edge(a, c, confidence=float(r['confidence']), lift=float(r['lift']), support=float(r['support']))
    pos = nx.spring_layout(G, seed=SEED, k=1.3)
    edge_x, edge_y = [], []
    mid_x, mid_y, mid_text = [], [], []
    for u, v, data in G.edges(data=True):
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
        mid_x.append((x0+x1)/2); mid_y.append((y0+y1)/2)
        mid_text.append(f'{u} → {v}<br>confidence={data["confidence"]:.3f}<br>lift={data["lift"]:.3f}<br>support={data["support"]:.4f}')
    edge_trace = go.Scatter(x=edge_x,y=edge_y,mode='lines',hoverinfo='skip',line=dict(width=1,color='#94A3B8'))
    mid_trace = go.Scatter(x=mid_x,y=mid_y,mode='markers',marker=dict(size=10,opacity=0.01),text=mid_text,hovertemplate='%{text}<extra></extra>')
    node_x,node_y,node_text,node_degree=[],[],[],[]
    for node in G.nodes():
        x,y=pos[node]; node_x.append(x); node_y.append(y); node_text.append(node); node_degree.append(G.degree(node))
    node_trace = go.Scatter(
        x=node_x,y=node_y,mode='markers+text',text=node_text,textposition='top center',
        marker=dict(size=[12+3*d for d in node_degree],color=node_degree,colorscale='Blues',showscale=True,colorbar=dict(title='Degree')),
        hovertemplate='%{text}<extra></extra>'
    )
    fig=go.Figure([edge_trace,mid_trace,node_trace])
    fig.update_layout(title='Cross-sell network — hover edges for rule evidence',showlegend=False,height=650,
                      xaxis=dict(visible=False),yaxis=dict(visible=False),margin=dict(l=10,r=10,t=55,b=10))
    return fig

fig_rule_network = build_rule_network(fp_rules_pd, 25)
fig_rule_network.show()
