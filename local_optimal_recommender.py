import ast
import math
import time

import pandas as pd
import numpy as np
from recommender import RecommenderHandler
from collections import defaultdict


class Local_Optimal_Recommender(RecommenderHandler):
    poi_occ = defaultdict(float)  # a poi placekey dictionary  that holds the avg occ of that POI

    def __init__(self, k, t, time_of_sim, dir_name, city):
        # initialize k and t, need to do once
        super().__init__(k, t, time_of_sim, dir_name, city)

        # create avg occ dict - need to do once
        self.__buid_poi_occ__()

        # extract top-k pois per user (top_k_pois), and num_users
        self.__extract_and_sort_pois__()

        # create weight matrix S and pointer I
        super().__weight_generator__(self.top_k_pois)

    def __extract_and_sort_pois__(self):
        """
        extracts and sorts the POIs per user randomly
        :return:
        """
        # queries = pd.read_csv('files/queries/'+self.dir_name+'/query_result_new_'+str(self.time_group)+'.csv')
        queries = super().get_queries()
        queries.drop(['latitude','longitude','radius','time'], axis=1, inplace=True)  # save on space
        queries['pois'] = queries.pois.apply(eval)
        self.num_users = len(queries)
        self.top_k_pois = []

        for i, query in queries.iterrows():
            # select up-to k pois based on local risk
            pois = np.array(query['pois'])
            occ = [self.poi_occ[x] for x in pois]

            # sort by occupancy
            dic = {pois[x]: occ[x] for x in range(len(pois))}
            dic = dict(sorted(dic.items(), key=lambda x: x[1]))

            tmp = list(dic.keys())
            if len(tmp) < self.K:
                self.top_k_pois.append(tmp)
            else:
                self.top_k_pois.append(tmp[:self.K])

    def __buid_poi_occ__(self):
        """
        construct a poi occupancy self.poi_occ
        :return:
        """
        for index, row in self.POIs.iterrows():
            minute = int(math.floor(self.time_group / 60))
            dwell_time = row['avg_dwell_time'] / 60
            dwell_time = dwell_time if dwell_time < self.max_dwell_time else self.max_dwell_time

            start = int(math.floor(self.time_of_sim + minute + (self.travel_time/60)))
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
        poi_dict = self.__update_occ__()

        for key, value in poi_dict.items():
            # updat poi occupancy after user redirection
            self.poi_occ[key] += value

    def rerun_top_k(self):
        # extract top-k pois per user (top_k_pois), and num_users
        self.__extract_and_sort_pois__()

        # create weight matrix S and pointer I
        super().__weight_generator__(self.top_k_pois)

    def save(self, save_as='local_optimal_occ'):
        self.__save_occ__(save_as)
