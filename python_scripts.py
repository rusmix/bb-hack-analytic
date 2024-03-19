import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import psycopg2
import sqlalchemy

import tqdm
from tqdm import tqdm
import datetime
import requests
import random

from numpy import random
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
import scipy 

import datetime
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from jupyterthemes import jtplot
jtplot.style('onedork')

import clickhouse_connect


# алгоритм матричной факторизации


from sklearn.metrics import mean_squared_error
class ExplicitMF:
    """
    Train a matrix factorization model using Alternating Least Squares
    to predict empty entries in a matrix
    
    Parameters
    ----------
    n_iters : int
        number of iterations to train the algorithm
        
    n_factors : int
        number of latent factors to use in matrix 
        factorization model, some machine-learning libraries
        denote this as rank
        
    reg : float
        regularization term for item/user latent factors,
        since lambda is a keyword in python we use reg instead
    """

    def __init__(self, n_iters, n_factors, reg):
        self.reg = reg
        self.n_iters = n_iters
        self.n_factors = n_factors  
        
    def fit(self, train):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings
        """
        self.n_user, self.n_item = train.shape
        self.user_factors = np.random.random((self.n_user, self.n_factors))
        self.item_factors = np.random.random((self.n_item, self.n_factors))

        self.train_mse_record = []   
        for _ in range(self.n_iters):
            self.user_factors = self._als_step(train, self.user_factors, self.item_factors)
            self.item_factors = self._als_step(train.T, self.item_factors, self.user_factors) 
            predictions = self.predict()
            train_mse = self.compute_mse(train, predictions)
            self.train_mse_record.append(train_mse)
        
        return self    
    
    def _als_step(self, ratings, solve_vecs, fixed_vecs):
        """
        when updating the user matrix,
        the item matrix is the fixed vector and vice versa
        """
        A = fixed_vecs.T.dot(fixed_vecs) + np.eye(self.n_factors) * self.reg
        b = ratings.dot(fixed_vecs)
        A_inv = np.linalg.inv(A)
        solve_vecs = b.dot(A_inv)
        return solve_vecs
    
    def predict(self):
        """predict ratings for every user and item"""
        pred = self.user_factors.dot(self.item_factors.T)
        return pred
    
    @staticmethod
    def compute_mse(y_true, y_pred):
        """ignore zero terms prior to comparing the mse"""
        mask = np.nonzero(y_true)
        mse = mean_squared_error(y_true[mask], y_pred[mask])
        return mse


# для отрисовки кривой обучения

def plot_learning_curve(model):
    """visualize the training/testing loss"""
    linewidth = 3
    plt.plot(model.train_mse_record, label = 'Train', linewidth = linewidth)
    plt.xlabel('iterations')
    plt.ylabel('MSE')
    plt.legend(loc = 'best')
    plt.show()


# забираем доступные казино игры


conn = psycopg2.connect(
    host = "localhost",
    database = "xaxaton_master",
    user = "postgres",
    password = "pass1",
    port = '5438')


games_df = pd.read_sql_query('''
select 
    game_id
from xaxaton_master.main.games
''', con = conn)


# начинаем генерацию данных по пользователям, определеяем кол-во игр в которые он сыграет с весами: 
# 50% от 1 до 3
# 40% от 5 до 10
# 10% от 10 до 50

# Дальше выбираем насколько каждая игра понравилась пользователю, также с весами:
# 40% 2-10 ставок
# 30% 10-50 ставок
# 20% 50-200 ставок
# 10% 200-500 ставок


test_data = pd.DataFrame(columns = ['transaction_id', 'gambler_id', 'partner_id', 'external_game_id', 'transaction_type', 'usd_amount', 'end_datetime'])

i = 1

for j in tqdm(range(1, 2000)):


    gg = random.random()
    games_cnt = [random.randint(1, 3), random.randint(5, 10), random.randint(10, 50)][0 if gg < 0.5 else 1 if gg < 0.9 else 2]

    for k in range(0, games_cnt):

        game_id = random.randint(1,934)

        ff = random.random()
        bets_cnt = [random.randint(2, 10), random.randint(10, 50), random.randint(50, 200), random.randint(200, 500)]\
            [0 if ff < 0.4 else 1 if ff < 0.7 else 2 if ff < 0.9 else 3]

        for ij in range(1, bets_cnt):

            test_data.loc[i, 'transaction_id'] = i
            test_data.loc[i, 'gambler_id'] = j
            test_data.loc[i, 'partner_id'] = 1
            test_data.loc[i, 'external_game_id'] = game_id
            test_data.loc[i, 'transaction_type'] = 'in'
            test_data.loc[i, 'usd_amount'] = random.randint(1, 100)
        
            if i == 1:
                
                test_data.loc[i, 'end_datetime'] = datetime(datetime.today().year
                                                         , datetime.today().month
                                                         , 1, 18, 0, 0)
        
            else:
        
                test_data.loc[i, 'end_datetime'] = test_data.loc[i-1, 'end_datetime'] + pd.Timedelta(seconds=1)
        
            i += 1



# аггрегируем данные для таблички на которой будет обучаться модель


df_agg = test_data.groupby(by=['gambler_id', 'external_game_id']).agg({'external_game_id': 'nunique', 'usd_amount': ['count', 'sum'], 'end_datetime': ['min', 'max']})
df_agg = df_agg.reset_index()
df_agg.columns = df_agg.columns.map('{0[0]}{0[1]}'.format) 
df_agg.rename(columns = {'external_game_idnunique': 'external_game_id_cnt', 'usd_amountcount': 'bet_cnt'
                         , 'usd_amountsum': 'usd_value', 'end_datetimemin': 'start_time', 'end_datetimemax': 'end_time'}, inplace = True)
df_agg.drop(columns = ['index', 'external_game_id_cnt'], inplace = True)
df_agg['time'] = (df_agg['end_time'] - df_agg['start_time'])
df_agg['time'] = df_agg['time'].apply(lambda x: x.total_seconds() / 60)
df_agg['time'] = df_agg['time'].apply(lambda x: 1 if x == 0 else x)


df_pivot = df_agg.pivot(index = 'gambler_id', columns = 'external_game_id', values = 'time')


# процесс нормализации данных с помощью логарифма

ubyi_norm = np.log(df_pivot)

ubyi_norm += abs(ubyi_norm.min().min())
print(ubyi_norm.min().min(), ubyi_norm.max().max())

ubyi_norm = ubyi_norm[(~ubyi_norm.isnull()).sum(axis=1) >= 3] # 3 - мин кол-во игр в которое нужно сыграть юзеру

ubyi_norm_0 = ubyi_norm.fillna(0)

# построение кривой обучения с различными параметрами

als_min_history = []
for i in [5,20,40,100,150,250,500]:
  als = ExplicitMF(n_iters = 100, n_factors = i, reg = 0.01)
  als.fit(ubyi_norm_0.to_numpy())
  plot_learning_curve(als)
  als_min_history.append(min(als.train_mse_record))


plt.plot([5,20,40,100,150,250,500], als_min_history)
plt.xlabel('Number of Factors')
plt.ylabel('MSE')

# создаем и обучаем модель

als = ExplicitMF(n_iters = 100, n_factors = 100, reg = 0.01)
als.fit(ubyi_norm_0.to_numpy())

lf_prod = np.matmul(als.user_factors, als.item_factors.T)
ubyi_mf = pd.DataFrame(lf_prod, index=ubyi_norm.index.values, columns=ubyi_norm.columns.values)


# разворачиваем матрицу в таблицу для postgresql

df_to_transport = ubyi_mf.unstack().reset_index()
df_to_transport.rename(columns = {'level_0': 'external_game_id', 'level_1': 'gambler_id', 0: 'value'}, inplace = True)


# отсылаем аггрегированные данные в БД аналитики

df_agg.to_sql('df_agg', schema = 'main', con = engine3, if_exists = 'append', index = False)
df_to_transport.to_sql('gamblers_preference', schema = 'main', con = engine3, if_exists = 'append', index = False)


# забираем данные по играм для фильтров

conn = psycopg2.connect(
    host = "localhost",
    database = "xaxaton_master",
    user = "postgres",
    password = "pass1",
    port = '5438')


games_df_temp = pd.read_sql_query('''
select 
	theme_id
    , tag_id
    , external_game_id as game_id
    , provider_id
    , partner_id
    , country
from xaxaton_master.main.games g 
left join xaxaton_master.main.games_tags gt 
	using(game_id)
left join xaxaton_master.main.games_themes gt2 
	using(game_id)
left join xaxaton_master.main.tags t 
	using(tag_id)
left join xaxaton_master.main.themes t2 
	using(theme_id)
left join xaxaton_master.main.partner_games pg 
	using(game_id)
left join xaxaton_master.main.partners p 
	using(partner_id)
left join xaxaton_master.main.providers_countries pc 
	using(provider_id)
''', con = conn)


# отсылаем данные для корректной работы фильтров

games_df_temp.to_sql('games', schema = 'main', con = engine3, if_exists = 'append', index = False)
