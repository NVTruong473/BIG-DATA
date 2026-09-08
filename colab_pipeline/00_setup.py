# Auto-generated from BIG_DATA_Customer_Intelligence_PRO_Colab.ipynb

# ---- source cell 3 ----
import os, math, time, json, shutil, warnings, itertools, statistics
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import psutil

from IPython.display import display, HTML, clear_output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score,
)
import umap.umap_ as umap

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator, RankingEvaluator
from pyspark.mllib.linalg import Vectors as OldVectors
from pyspark.mllib.linalg.distributed import RowMatrix

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 120)
pio.templates.default = 'plotly_white'

SEED = 42
QUALITY_MODE = 'max'       # 'balanced' hoặc 'max'
TOP_K = 10
TEST_FRAC = 0.20
MIN_USER_RATINGS_FOR_TEST = 5
RELEVANCE_THRESHOLD = 4.0
STAT_SEEDS = [42, 52, 62] if QUALITY_MODE == 'max' else [42, 52]
OUTPUT_DIR = Path('/content/big_data_outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THEME = {
    'navy': '#0F172A', 'blue': '#2563EB', 'cyan': '#0891B2',
    'green': '#059669', 'amber': '#D97706', 'red': '#DC2626',
    'muted': '#64748B', 'card': '#F8FAFC'
}

spark = (
    SparkSession.builder
    .appName('BIG-DATA-Customer-Intelligence')
    .master('local[*]')
    .config('spark.driver.memory', '6g')
    .config('spark.sql.shuffle.partitions', '8')
    .config('spark.sql.adaptive.enabled', 'true')
    .config('spark.sql.adaptive.coalescePartitions.enabled', 'true')
    .getOrCreate()
)
sc = spark.sparkContext
sc.setLogLevel('WARN')
print('Spark:', spark.version, '| defaultParallelism:', sc.defaultParallelism)

# ---- source cell 5 ----
try:
    import ipywidgets as widgets
    EXEC_OUT = widgets.Output()
    display(EXEC_OUT)
    with EXEC_OUT:
        display(HTML("""
        <div style='padding:22px;border:1px solid #e2e8f0;border-radius:16px;background:#f8fafc'>
          <b>Executive Dashboard</b><br>
          <span style='color:#64748b'>Run all cells. Dashboard sẽ tự refresh khi toàn bộ pipeline hoàn tất.</span>
        </div>
        """))
except Exception:
    EXEC_OUT = None
    display(HTML('<b>Executive Dashboard:</b> sẽ được render ở phần cuối notebook.'))

# ---- source cell 7 ----
import urllib.request

BASE_RAW = 'https://raw.githubusercontent.com/NVTruong473/BIG-DATA/main/end/end/'
DATASETS = {
    'baskets.csv': BASE_RAW + 'baskets.csv',
    'ratings2k.csv': BASE_RAW + 'ratings2k.csv',
}

for name, url in DATASETS.items():
    path = Path('/content') / name
    if not path.exists() or path.stat().st_size == 0:
        print('Downloading', name)
        urllib.request.urlretrieve(url, path)
    print(name, f'{path.stat().st_size/1024:.1f} KB')

BASKETS_PATH = '/content/baskets.csv'
RATINGS_PATH = '/content/ratings2k.csv'

# ---- source cell 8 ----
baskets_raw = spark.read.option('header', True).option('inferSchema', True).csv(BASKETS_PATH)
ratings_df = (
    spark.read.option('header', True).option('inferSchema', True).csv(RATINGS_PATH)
    .select(
        F.col('user').cast('int').alias('user'),
        F.col('item').cast('int').alias('item'),
        F.col('rating').cast('double').alias('rating')
    )
    .dropna()
)

# Parse nhiều date format phổ biến để notebook robust hơn.
baskets_df = (
    baskets_raw
    .withColumn(
        'date_parsed',
        F.coalesce(
            F.to_date('Date', 'dd-MM-yyyy'),
            F.to_date('Date', 'dd/MM/yyyy'),
            F.to_date('Date', 'yyyy-MM-dd')
        )
    )
    .filter(F.col('Member_number').isNotNull() & F.col('itemDescription').isNotNull())
)

print('baskets rows =', baskets_df.count())
print('ratings rows =', ratings_df.count())
baskets_df.printSchema()
ratings_df.printSchema()

# ---- source cell 10 ----
ratings_rows = ratings_df.count()
n_users = ratings_df.select('user').distinct().count()
n_items = ratings_df.select('item').distinct().count()
rating_min, rating_max, rating_mean = ratings_df.agg(
    F.min('rating'), F.max('rating'), F.avg('rating')
).first()
sparsity = 1.0 - ratings_rows / max(1, n_users * n_items)

basket_rows = baskets_df.count()
n_members = baskets_df.select('Member_number').distinct().count()
n_baskets = baskets_df.select('Member_number', 'Date').distinct().count()
n_products = baskets_df.select('itemDescription').distinct().count()
null_dates = baskets_df.filter(F.col('date_parsed').isNull()).count()

DATA_KPIS = {
    'Rating rows': ratings_rows,
    'Users': n_users,
    'Items': n_items,
    'Rating sparsity': sparsity,
    'Basket rows': basket_rows,
    'Members': n_members,
    'Distinct baskets': n_baskets,
    'Products': n_products,
    'Unparsed dates': null_dates,
}

display(pd.DataFrame([DATA_KPIS]))
print(f'Rating scale: {rating_min} → {rating_max}; mean={rating_mean:.3f}; sparsity={sparsity:.2%}')

# ---- source cell 11 ----
# Visual health check
rating_dist_pd = ratings_df.groupBy('rating').count().orderBy('rating').toPandas()
fig_rating_dist = px.bar(
    rating_dist_pd, x='rating', y='count',
    title='Rating distribution',
    labels={'rating':'Rating', 'count':'Number of ratings'}
)
fig_rating_dist.update_traces(hovertemplate='Rating=%{x}<br>Count=%{y}<extra></extra>')
fig_rating_dist.show()
