import copy
import math

import numpy as np
import pandas as pd
import ast
from collections import defaultdict


class RecommenderHandler:
    POIs = None          # a dataframe of all POIs and their relevant info
    top_k_pois = None    # selected pois per user
    S = None             # weights distribution info for k between [0,k]
    I = None             # each user points to a value in S
    K = None             # number of recommendations
    time_group = None    # current time
    num_users = None     # number of users at each time_group

    # paramters of simulation and assumptions
    time_of_sim = None     # the time the simulation begins
    max_dwell_time = 3   # apply a max of dwell time per POI
    travel_time = 20     # assume a 20 min travel time from querying
    dir_name = None

    city = None

    def __init__(self, k, t, time_of_sim, dir_name, city):
        self.dir_name = dir_name
        self.K = k
        self.time_group = t
        self.city = city
        self.time_of_sim = time_of_sim
        self.__load_occupancies__()

    def __load_occupancies__(self):
        df = pd.read_csv('files/ds/'+self.city+'/extrapolated_visits.csv')
        df['avg_visits'] = df.avg_visits.apply(lambda x: ast.literal_eval(x))
        self.POIs = df

    def __get_vals__(self, s):
        """
        :param s: number of eligible POIs for a user
        :return: the weight distribution based on s
        """
        if s == 1:
            res = [0]*self.K
            res[0] = 1
            return res
        if s < 1:
            res = [0]*self.K
            return res

        sum_val = 0
        res = [0] * self.K
        tmp = [0] * self.K

        # get the sum
        for index in range(1, s + 1):
            sum_val += np.power((1 - 1 / s), (s * index))
            tmp[index - 1] = index - 1
        # get weights
        for index in range(1, s + 1):
            res[index - 1] = np.power((1 - 1 / s), (s * index)) / sum_val
            if np.isnan(res[index - 1]): res[index - 1] = 0

        return res

    def __weight_generator__(self, func):
        """
        :param num_users: number of users in current time-group
        :param func: relevant list that helps determine the number of users
        :return: None
        """
        self.S, self.I = [], []
        # compute matrix S (weight dist from [0,k])
        for tmp_k in range(self.K + 1):
            # first row is irrelevant
            res = self.__get_vals__(tmp_k)
            self.S.append(res)
        self.S = np.array(self.S)

        # compute pointer for each user to S
        for index in range(self.num_users):
            s = int(min(len(func[index]), self.K))
            self.I.append(int(s))
        self.I = np.array(self.I, dtype=int)

    def get_I(self):
        return self.I

    def get_S(self):
        return self.S

    def get_POIs(self):
        return self.POIs

    def get_top_k(self):
        return self.top_k_pois

    def __update_occ__(self):
        """
        update POIs occupancies based on the top_k_pois selected and weight of each selection
        :return:
        """
        poi_dict = defaultdict(float)

        for user_index in range(len(self.top_k_pois)):
            for poi_place in range(len(self.top_k_pois[user_index])):
                poi_dict[self.top_k_pois[user_index][poi_place]] += self.S[self.I[user_index], poi_place]

        # update occupancy
        begin_time = self.time_of_sim + ((self.time_group+1) * 2 / 3600) + (self.travel_time/60)
        tmp1 = copy.deepcopy(self.POIs['avg_visits'].tolist())
        for key, value in poi_dict.items():
            dwell_time = self.POIs[self.POIs['placekey'] == key]['avg_dwell_time'].tolist()[0] / 60
            dwell_time = math.ceil(dwell_time) if dwell_time < self.max_dwell_time else self.max_dwell_time
            self.POIs[self.POIs['placekey'] == key]['avg_visits'].tolist()[0][int(math.floor(begin_time)):int(math.ceil(begin_time + dwell_time)+1)] = \
                list(np.array(self.POIs[self.POIs['placekey'] == key]['avg_visits'].tolist()[0][int(math.floor(begin_time)):int(math.ceil(begin_time + dwell_time)+1)]) + value)
        tmp2 = copy.deepcopy(self.POIs['avg_visits'].tolist())
        return poi_dict

    def __save_occ__(self, type):
        """
        saves the new occupancy
        :param type: dist_occ / random_occ / optimal_occ / local optimal_occ
        :return: None
        """
        df = pd.read_csv('files/ds/'+self.city+'/extrapolated_visits.csv')
        if type in df.columns:
            df = df.drop([type], axis=1)

        self.POIs[type] = self.POIs.avg_visits
        df = pd.merge(df, self.POIs[['placekey',type]], on='placekey')
        df.to_csv('files/ds/'+self.city+'/extrapolated_visits.csv', index=False)
        self.POIs = self.POIs.drop([type], axis=1)

    def update_time_group(self, time_group):
        self.time_group = time_group

    def get_queries(self):
        queries = pd.read_csv('files/ds/'+self.city+'/queries/'+self.dir_name+'/query_result_' +
                              str(self.time_group) + '.csv')
        return queries