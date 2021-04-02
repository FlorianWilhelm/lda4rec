#!/usr/bin/env python
# coding: utf-8

# In[1]:


import functools
import itertools
import logging
import math
import os
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import yaml

# get_ipython().run_line_magic("load_ext", "autoreload")
# get_ipython().run_line_magic("autoreload", "2")
#
# get_ipython().run_line_magic("matplotlib", "inline")
# get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")

sns.set_context("poster")
sns.set(rc={"figure.figsize": (16, 12.0)})
sns.set_style("whitegrid")

import numpy as np
import pandas as pd
import torch.nn.functional as F

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


# In[2]:


import neptune
import pyro
import pyro.distributions as dist
import pyro.optim as optim
import torch
from pyro.distributions import constraints
from pyro.infer import SVI, Predictive, Trace_ELBO, TraceEnum_ELBO, config_enumerate

from lda4rec.datasets import DataLoader, Interactions, random_train_test_split
from lda4rec.estimators import LDA4RecEst, MFEst, PopEst, SNMFEst
from lda4rec.evaluations import auc_score, mrr_score, precision_recall_score, summary
from lda4rec.utils import cmp_ranks, process_ids

# In[3]:


# In[4]:


# init dummy neptune to avoid problems with logging
neptune.init("a/b", backend=neptune.OfflineBackend())


# In[5]:


from icecream import ic, install

install()
# configure icecream
def ic_str(obj):
    if hasattr(obj, "shape"):
        return f"{obj} "  #
    else:
        return str(obj)


# In[6]:


ic.configureOutput(argToStringFunction=ic_str)


# ## Experimenting with different estimators

# In[7]:


loader = DataLoader()
data = loader.load_movielens("100k")


# In[8]:


max_interactions = 200
data.max_user_interactions_(max_interactions)


# In[9]:


data.implicit_(0.0)
train, test = random_train_test_split(data)


# In[10]:


pop_est = PopEst()
pop_est.fit(train)


# In[11]:


df = summary(pop_est, train=train, test=test)
print(df)

# In[26]:


mf_est = MFEst(embedding_dim=4, n_iter=10)
final_loss = mf_est.fit(train)
print(final_loss)

# In[29]:


df = summary(mf_est, train=train, test=test)
print(df)

# In[16]:


lda_est = LDA4RecEst(
    embedding_dim=4, n_iter=10_000, batch_size=128, learning_rate=0.001, use_jit=True
)


# In[17]:


final_loss = lda_est.fit(train)


print(final_loss)


df = summary(lda_est, train=train, test=test)

print(df)

# In[84]:
