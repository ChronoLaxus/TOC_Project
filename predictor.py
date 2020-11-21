import tensorflow as tf
from model import Model
from input_pipe import InputPipe
from feeder import VarFeeder
from tqdm import trange
import matplotlib.pyplot as plt
import collections
import pandas as pd
import numpy as np
from trainer import predict
from hparams import build_hparams
import hparams
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

def smape(true, pred):
    summ = np.abs(true) + np.abs(pred)
    smape = np.where(summ == 0, 0, np.abs(true - pred) / summ)
    return smape * 200

def mean_smape(true, pred):
    raw_smape = smape(true, pred)
    masked_smape = np.ma.array(raw_smape, mask=np.isnan(raw_smape))
    return masked_smape.mean()

from make_features import read_all
df_all = read_all()
df_all.columns

prev = df_all

paths = [p for p in tf.train.get_checkpoint_state('data/cpt/s32').all_model_checkpoint_paths]

t_preds = []
for tm in range(3):
    tf.reset_default_graph()
    t_preds.append(predict(paths, build_hparams(hparams.params_s32), back_offset=0, predict_window=63,
                    n_models=3, target_model=tm, seed=2, batch_size=2048, asgd=True))

preds=sum(t_preds) /3

missing_pages = prev.index.difference(preds.index)
# Use zeros for missing pages
rmdf = pd.DataFrame(index=missing_pages,
                    data=np.tile(0, (len(preds.columns),len(missing_pages))).T, columns=preds.columns)
f_preds = preds.append(rmdf).sort_index()

# Use zero for negative predictions
f_preds[f_preds < 0.5] = 0
# Rouns predictions to nearest int
f_preds = np.round(f_preds).astype(np.int64)


page="Heahmund_en.wikipedia.org_desktop_all-agents"
prev.loc[page].fillna(0).plot(logy=True)

#gt.loc[page].fillna(0).plot(logy=True)
f_preds.loc[page].plot(logy=True)


def read_keys():
    import os.path
    key_file = 'data/keys2.pkl'
    if os.path.exists(key_file):
        return pd.read_pickle(key_file)
    else:
        print('Reading keys...')
        raw_keys = pd.read_csv('data/key_2.csv.zip')
        print('Processing keys...')
        pagedate = raw_keys.Page.str.rsplit('_', expand=True, n=1).rename(columns={0:'page',1:'date_str'})
        keys = raw_keys.drop('Page', axis=1).assign(page=pagedate.page, date=pd.to_datetime(pagedate.date_str))
        del raw_keys, pagedate
        print('Pivoting keys...')
        pkeys = keys.pivot(index='page', columns='date', values='Id')
        print('Storing keys...')
        pkeys.to_pickle(key_file)
        return pkeys
keys = read_keys()

subm_preds = f_preds.loc[:, '2017-09-13':]
assert np.all(subm_preds.index == keys.index)
assert np.all(subm_preds.columns == keys.columns)
answers = pd.DataFrame({'Id':keys.values.flatten(), 'Visits':np.round(subm_preds).astype(np.int64).values.flatten()})
answers.to_csv('data/submission.csv.gz', compression='gzip', index=False, header=True)


f_preds