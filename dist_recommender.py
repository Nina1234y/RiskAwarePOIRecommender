import time

import pandas as pd
import numpy as np
from recommender import RecommenderHandler


class Dist_Recommender(RecommenderHandler):

    def __init__(self, k, t, time_of_sim, dir_name, city):
        # initialize k and t
        super().__init__(k, t, time_of_sim, dir_name, city)

        # extract top-k pois per user (top_k_pois), and num_users
        self.__extract_and_sort_pois__()

        # create weight matrix S and pointer I
        super().__weight_generator__(self.top_k_pois)

    def __extract_and_sort_pois__(self):
        """
        extracts distances and sorts the POIs per user based on distance
        (closest first, furthest last)
        :return:
        """
        # queries = pd.read_csv('files/queries/'+self.dir_name+'/query_result_new_'+str(self.time_group)+'.csv')
        queries = super().get_queries()
        queries.drop(['latitude','longitude','radius','time'], axis=1, inplace=True)  # save on space
        queries['pois'] = queries.pois.apply(eval)
        queries['dists'] = queries.dists.apply(eval)
        self.num_users = len(queries)
        self.top_k_pois = []

        counter = 0
        for i, query in queries.iterrows():
            pois = np.array(query['pois'])
            d = np.array(query['dists'])

            # sort by distance
            dic = {pois[x]: d[x] for x in range(len(pois))}
            dic = dict(sorted(dic.items(), key=lambda x: x[1]))

            # select up-to k pois based on distances
            tmp = list(dic.keys())
            if len(tmp) < self.K:
                self.top_k_pois.append(tmp)
            else:
                self.top_k_pois.append(tmp[:self.K])
            counter += 1

    def rerun_top_k(self):
        # extract top-k pois per user (top_k_pois), and num_users
        self.__extract_and_sort_pois__()

        # create weight matrix S and pointer I
        super().__weight_generator__(self.top_k_pois)

    def save(self,save_as='dist_occ'):
        self.__save_occ__(save_as)

    def update_occ(self):
        self.__update_occ__()
