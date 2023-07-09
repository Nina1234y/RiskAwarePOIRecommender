import copy
import itertools
import math
import time

import pandas as pd
import numpy as np
from recommender import RecommenderHandler
from assignment_models import ordered_many_to_many_assignment
from collections import defaultdict


class Ordered_Global_Optimal_Recommender(RecommenderHandler):
    poi_occ = defaultdict(float)  # a poi placekey dictionary  that holds the avg occ of that POI
    pois_keys = None  # a list of placekeys to reference the current POIs in the "recommendation round"
    D = None  # eligibility matrix (user i can go to poi j)
    A = None  # list of geo area of current pois
    res = None  # recommendation results

    # parameters
    X = 16  # social distancing area

    def __init__(self, k, t, time_of_sim, dir_name, city):
        # initialize k and t, need to do once
        super().__init__(k, t, time_of_sim, dir_name, city)

        self.POIs.sort_values(by=['placekey'], ignore_index=True, inplace=True)

        # create avg occ dict - need to do once
        self.__buid_poi_occ__()

        # extract top-k pois per user (top_k_pois), and num_users
        N, M, occ = self.__extract_and_sort_pois__()

        # get optimal recommendation
        self.look_for_recommendations(N, M, occ)

    def look_for_recommendations(self, N, M, occ):
        self.res = ordered_many_to_many_assignment(N, M, occ, self.D, self.K, self.S, self.A, self.X, self.I)

    def __construct_eligibility_matrix__(self, queries):
        start = time.time()
        self.D = np.zeros((len(queries), len(self.pois_keys)), dtype=int)
        counter = 0

        for i, query in queries.iterrows():
            pois = np.array(query['pois'])
            indices = [np.where(self.pois_keys == x)[0][0] for x in pois]
            for index in indices:
                self.D[counter, index] = 1
            counter += 1
        end = time.time()
        # print('time took to construct eligibility matrix: ' + str((end - start) / 60) + ' min')

    def __get_area__(self):
        self.A = self.POIs[self.POIs['placekey'].isin(self.pois_keys)]['geodesic_area'].tolist()

    def __extract_and_sort_pois__(self):
        """
        extracts and sorts the POIs per user randomly
        :return:
        """
        # queries = pd.read_csv('files/queries/'+self.dir_name+'/query_result_new_'+str(self.time_group)+'.csv')
        queries = super().get_queries()
        queries.drop(['latitude', 'longitude', 'radius', 'time'], axis=1, inplace=True)  # save on space
        queries['pois'] = queries.pois.apply(eval)
        self.num_users = len(queries)
        self.top_k_pois = []
        pois = sorted(list(set(itertools.chain(*queries['pois']))))
        self.pois_keys = np.array(pois)

        # construct eligibility matrix
        self.__construct_eligibility_matrix__(queries)

        # get area A
        self.__get_area__()

        # create weight matrix S and pointer I
        tmp_func = [[1] * sum(x) for x in self.D]
        super().__weight_generator__(tmp_func)

        # extract current poi occ
        occ = np.array([self.poi_occ[x] for x in self.pois_keys])

        N = len(queries)
        M = len(pois)
        return N, M, occ

    def __buid_poi_occ__(self):
        """
        construct a poi occupancy self.poi_occ
        This build to the all pois, not just ones in current round
        :return:
        """
        for index, row in self.POIs.iterrows():
            minute = int(math.floor(self.time_group / 60))
            dwell_time = row['avg_dwell_time'] / 60
            dwell_time = dwell_time if dwell_time < self.max_dwell_time else self.max_dwell_time

            start = int(math.floor(self.time_of_sim + minute + (self.travel_time / 60)))
            end = start + dwell_time
            tmp_res = row['avg_visits'][start] * (60 - minute)
            accumulator = (60 - minute)

            while np.floor(end) > start:
                start += 1

                if end == start:
                    tmp_res += row['avg_visits'][start] * (end - np.floor(end)) * 60
                    accumulator += (end - np.floor(end)) * 60
                else:
                    tmp_res += row['avg_visits'][start] * 60
                    accumulator += 60

            self.poi_occ[row['placekey']] = (tmp_res / accumulator)

    def update_occ(self):
        """
        update POIs occupancies based on the top_k_pois selected and weight of each selection
        :return:
        """
        a = self.res.find_matching_vars(pattern="y_")
        begin_time = self.time_of_sim + ((self.time_group + 1) * 2 / 3600) + (self.travel_time / 60)

        tmp1 = copy.deepcopy(self.POIs['avg_visits'].tolist())
        for fv in a:
            if fv.solution_value > 0:
                value = fv.solution_value
                cell = [int(x) for x in fv.name[2:].split('_')]
                key = self.pois_keys[cell[1]]
                self.poi_occ[key] += value

                dwell_time = self.POIs[self.POIs['placekey'] == key]['avg_dwell_time'].tolist()[0] / 60
                dwell_time = math.ceil(dwell_time) if dwell_time < self.max_dwell_time else self.max_dwell_time
                self.POIs[self.POIs['placekey'] == key]['avg_visits'].tolist()[0][
                int(math.floor(begin_time)):int(math.ceil(begin_time + dwell_time) + 1)] = \
                    list(np.array(self.POIs[self.POIs['placekey'] == key]['avg_visits'].tolist()[0][
                                  int(math.floor(begin_time)):int(math.ceil(begin_time + dwell_time) + 1)]) + value)
        tmp2 = copy.deepcopy(self.POIs['avg_visits'].tolist())

        if tmp1 == tmp2:
            print('tmp1 == tmp2: ' + str(tmp1 == tmp2))

        # update user recommendation
        a = self.res.find_matching_vars(pattern="Q_")
        self.top_k_pois = []
        user_rec = dict()

        for fv in a:
            if fv.solution_value > 0:

                cell = [int(x) for x in fv.name[2:].split('_')]
                user = cell[0]
                poi = cell[1]
                rank = cell[2]
                if (user in user_rec.keys()):
                    user_rec[user][rank] = poi
                else:
                    user_rec[user] = [0]*self.K
                    user_rec[user][rank] = poi

        self.top_k_pois = list(user_rec.values())

    def rerun_top_k(self):
        # extract top-k pois per user (top_k_pois), and num_users
        N, M, occ = self.__extract_and_sort_pois__()

        # get optimal recommendation
        self.look_for_recommendations(N, M, occ)

    def save(self, save_as='global_optimal_occ'):
        self.__save_occ__(save_as)

    def get_recommendation(self):
        return self.res
