# Auto-generated from BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

# ---- source cell 64 ----
# Compute evidence-based narratives.
conclusions=[]
conclusions.append(
    f"SVD selected {selected_svd_k} latent dimensions, capturing {cum_energy[selected_svd_k-1]:.1%} of centered-matrix energy."
)
conclusions.append(
    f"CURE selected k={int(best_cure['k'])}, representatives={int(best_cure['representatives'])}, compression={best_cure['compression']:.2f}; "
    f"final silhouette={cure_sil:.3f}, stability ARI={cure_stability_ari:.3f}."
)
conclusions.append(
    f"Recommendation champion = {champion['model']} with RMSE={champion['RMSE_mean']:.3f}, MAE={champion['MAE_mean']:.3f}, "
    f"Precision@{TOP_K}={champion['Precision_mean']:.3f}, Recall@{TOP_K}={champion['Recall_mean']:.3f}."
)
if len(fp_rules_pd):
    br=fp_rules_pd.iloc[0]
    conclusions.append(
        f"Top actionable market-basket rule: {br['antecedent_text']} → {br['consequent_text']} "
        f"(confidence={br['confidence']:.3f}, lift={br['lift']:.3f}, support={br['support']:.4f})."
    )

common=scaling_pd.groupby('rows').filter(lambda g: set(g.engine)=={'Spark','pandas'})
if len(common):
    largest=common.rows.max(); comp=common[common.rows==largest].set_index('engine')
    if comp.loc['Spark','runtime_sec'] < comp.loc['pandas','runtime_sec']:
        conclusions.append(f"At {int(largest):,} rows in this local benchmark, Spark is faster than pandas for the tested aggregation.")
    else:
        conclusions.append(f"At {int(largest):,} rows in this local benchmark, pandas remains faster; Spark's demonstrated value here is distributed execution semantics, scaling path, physical-plan visibility and cache reuse—not a fabricated single-node speed claim.")
conclusions.append(f"Spark cache speedup for the repeated aggregation at scale {cache_scale}× = {cache_speedup:.2f}×.")

for i,c in enumerate(conclusions,1): print(f'{i}. {c}')

# ---- source cell 66 ----
def fmt_num(x,kind='num'):
    if kind=='pct': return f'{x:.1%}'
    if isinstance(x,(float,np.floating)): return f'{x:.3f}'
    return f'{int(x):,}' if isinstance(x,(int,np.integer)) else str(x)

def kpi_card(label,value,tooltip):
    return f"""
    <div title="{tooltip}" style='background:white;border:1px solid #E2E8F0;border-radius:14px;padding:14px 16px;min-width:155px;box-shadow:0 2px 10px rgba(15,23,42,.05)'>
      <div style='font-size:12px;color:#64748B;font-weight:600'>{label} ⓘ</div>
      <div style='font-size:24px;color:#0F172A;font-weight:750;margin-top:4px'>{value}</div>
    </div>"""

cards=''.join([
    kpi_card('Ratings',fmt_num(ratings_rows),'Observed user-item ratings.'),
    kpi_card('Users',fmt_num(n_users),'Unique users in ratings2k.'),
    kpi_card('Sparsity',fmt_num(sparsity,'pct'),'1 - observed ratings / all possible user-item pairs.'),
    kpi_card('Model champion',str(champion['model']),'Balanced score across RMSE, MAE, Precision@10 and Recall@10.'),
    kpi_card('RMSE',fmt_num(champion['RMSE_mean']),'Lower is better. Mean across statistical-validation seeds.'),
    kpi_card(f'Precision@{TOP_K}',fmt_num(champion['Precision_mean']),'Relevant recommendations divided by K.'),
    kpi_card(f'Recall@{TOP_K}',fmt_num(champion['Recall_mean']),'Share of held-out relevant items recovered in Top-K.'),
    kpi_card('SVD dimensions',fmt_num(selected_svd_k),f'Chosen by captured energy; {cum_energy[selected_svd_k-1]:.1%} captured.'),
    kpi_card('CURE k',fmt_num(int(best_cure['k'])),'Selected by silhouette, DB, CH and balance composite.'),
    kpi_card('CURE silhouette',fmt_num(cure_sil),'Higher indicates stronger separation/cohesion in latent space.'),
    kpi_card('CURE stability ARI',fmt_num(cure_stability_ari),'Agreement of cluster assignments across repeated samples; 1 is perfect.'),
    kpi_card('Spark cache speedup',f'{cache_speedup:.2f}×','Repeated aggregation after cache materialization vs uncached run.'),
])

dashboard_html=f"""
<div style='font-family:Inter,Arial,sans-serif;background:#F8FAFC;border-radius:18px;padding:20px;border:1px solid #E2E8F0'>
  <div style='display:flex;justify-content:space-between;align-items:end;margin-bottom:14px'>
    <div><div style='font-size:12px;color:#2563EB;font-weight:700;letter-spacing:.08em'>EXECUTIVE VIEW</div>
    <div style='font-size:25px;font-weight:800;color:#0F172A'>Big Data Customer Intelligence</div></div>
    <div style='font-size:12px;color:#64748B'>Hover KPI titles / charts for evidence</div>
  </div>
  <div style='display:flex;gap:10px;flex-wrap:wrap'>{cards}</div>
</div>
"""

display(HTML(dashboard_html))
fig_model_quality.show()
fig_cluster_umap.show()
fig_rules.show()
fig_scaling.show()

if EXEC_OUT is not None:
    try:
        with EXEC_OUT:
            clear_output(wait=True)
            display(HTML(dashboard_html))
            display(fig_model_quality)
            display(fig_cluster_umap)
            display(fig_rules)
            display(fig_scaling)
    except Exception as e:
        print('Top dashboard refresh skipped:',e)

# ---- source cell 68 ----
# Tables
rating_stats.to_csv(OUTPUT_DIR/'recommendation_rating_metrics.csv',index=False)
ranking_stats.to_csv(OUTPUT_DIR/'recommendation_ranking_metrics.csv',index=False)
model_score.to_csv(OUTPUT_DIR/'recommendation_model_champion.csv',index=False)
cure_tuning_pd.to_csv(OUTPUT_DIR/'cure_tuning.csv',index=False)
cluster_profile.to_csv(OUTPUT_DIR/'cluster_profiles.csv',index=False)
rfv_pd.to_csv(OUTPUT_DIR/'rfv_customer_segments.csv',index=False)
fp_tuning_pd.to_csv(OUTPUT_DIR/'fpgrowth_tuning.csv',index=False)
fp_rules_pd.to_csv(OUTPUT_DIR/'fpgrowth_top_rules.csv',index=False)
scaling_pd.to_csv(OUTPUT_DIR/'spark_scalability.csv',index=False)
svd_curve.to_csv(OUTPUT_DIR/'svd_energy_curve.csv',index=False)

# CURE representative points
rep_rows=[]
for cid,reps in enumerate(cure_reps):
    for rid,p in enumerate(reps):
        rep_rows.append({'cluster':cid,'representative':rid,**{f'z{i+1}':float(v) for i,v in enumerate(p)}})
pd.DataFrame(rep_rows).to_csv(OUTPUT_DIR/'cure_representatives.csv',index=False)

# Spark ALS model from first statistical-validation seed.
try:
    first_als_model.write().overwrite().save(str(OUTPUT_DIR/'als_model'))
except Exception as e:
    print('ALS export warning:',e)

# One HTML executive report.
parts=["<html><head><meta charset='utf-8'><title>Big Data Executive Dashboard</title></head><body style='font-family:Arial;background:#f8fafc;padding:24px'>",dashboard_html]
figures=[fig_model_quality,fig_cluster_umap,fig_cluster_radar,fig_rules,fig_rule_network,fig_rfv,fig_svd,fig_scaling,fig_throughput]
for i,fig in enumerate(figures):
    parts.append(pio.to_html(fig,full_html=False,include_plotlyjs='cdn' if i==0 else False))
parts.append('<h2>Automatic conclusions</h2><ol>'+''.join(f'<li>{c}</li>' for c in conclusions)+'</ol></body></html>')
(OUTPUT_DIR/'executive_dashboard.html').write_text('\n'.join(parts),encoding='utf-8')

# Save conclusions JSON for reproducibility.
(OUTPUT_DIR/'summary.json').write_text(json.dumps({
    'data_kpis':DATA_KPIS,
    'selected_svd_k':selected_svd_k,
    'cure':{k:(float(v) if isinstance(v,(np.floating,float)) else int(v) if isinstance(v,(np.integer,int)) else v) for k,v in best_cure.items() if k in ['k','representatives','compression','silhouette','davies_bouldin','calinski_harabasz']},
    'champion':{k:(float(v) if isinstance(v,(np.floating,float)) else v) for k,v in champion.to_dict().items()},
    'conclusions':conclusions
},indent=2,ensure_ascii=False),encoding='utf-8')

zip_path=shutil.make_archive('/content/BIG_DATA_Customer_Intelligence_Outputs','zip',OUTPUT_DIR)
print('Exported:',zip_path)
print('HTML dashboard:',OUTPUT_DIR/'executive_dashboard.html')
print('All artifacts:',sorted(p.name for p in OUTPUT_DIR.iterdir()))

# ---- source cell 69 ----
# Optional Colab downloads — uncomment when needed.
# from google.colab import files
# files.download('/content/BIG_DATA_Customer_Intelligence_Outputs.zip')
# files.download('/content/big_data_outputs/executive_dashboard.html')
