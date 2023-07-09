import time

import pandas as pd
import numpy as np
from recommender import RecommenderHandler


class Random_Recommender(RecommenderHandler):

    def __init__(self, k, t, time_of_sim, dir_name, city):
        # initialize k and t
        super().__init__(k, t, time_of_sim, dir_name, city)

        # extract top-k pois per user (top_k_pois), and num_users
        self.__extract_and_sort_pois__()

        # create weight matrix S and pointer I
        super().__weight_generator__(self.top_k_pois)

    def __extract_and_sort_pois__(self):
        """
        extracts and sorts the POIs per user randomly
        :return:
        """
        #queries = pd.read_csv('files/ds/'+self.city+'/queries/'+self.dir_name+'/query_result_' + str(self.time_group) + '.csv')
        queries = super().get_queries()
        queries.drop(['latitude', 'longitude', 'radius', 'time'], axis=1, inplace=True)  # save on space
        queries['pois'] = queries.pois.apply(eval)
        self.num_users = len(queries)
        self.top_k_pois = []

        counter = 0
        for i, query in queries.iterrows():
            # select up-to k pois based on distances
            if len(query['pois']) < self.K:
                self.top_k_pois.append(np.random.choice(query['pois'], size=len(query['pois'])))
            else:
                self.top_k_pois.append(np.random.choice(query['pois'], size=self.K))
            counter += 1

    def rerun_top_k(self):
        # extract top-k pois per user (top_k_pois), and num_users
        self.__extract_and_sort_pois__()

        # create weight matrix S and pointer I
        super().__weight_generator__(self.top_k_pois)

    def save(self,save_as='random_occ'):
        self.__save_occ__(save_as)

    def update_occ(self):
        self.__update_occ__()
