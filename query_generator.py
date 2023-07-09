"""
This file generates queries that simulate user request over DUR seconds for CATEGORY category.
Each timestamp is treated as a second, where it can get MIN_Q to MAX_Q user requests.
The results are POIs referring to the appropriate category, within MAX_RAD Km.
It saves the results to SAVE_TO file.
"""

import pandas as pd
import numpy as np
import os
import itertools


def haversine_dist(source, latitude, longitude):
    """
    :param source: point of origin
    :param latitude:  destination's latitude
    :param longitude: destination's longitude
    :return: haversine distance in Km
    """
    R = 6373.0
    lat1 = np.deg2rad(source[0])
    lon1 = np.deg2rad(source[1])
    lat2 = np.deg2rad(latitude)
    lon2 = np.deg2rad(longitude)
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    d = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    cons = 2 * np.arctan2(np.sqrt(d), np.sqrt(1 - d))
    return R * cons


def get_queries(bounds_1, bounds_2, Q: int, dur: int, radius: int, SAVE_TO: str, FILE:str):
    """
    :param bounds_1: longitudue and latitude 1st bound
    :param bounds_2: longitudue and latitude 2nd bound
    :param Q: number of queries at each timestamp
    :param dur: length of the simulation in seconds
    :param radius: maximum search radius within bounds
    :param SAVE_TO: redirect query output to file SAVE_TO
    :param FILE: input file
    :return: creates a csv file containing the source (within the bounds),
             search radius,
             search time, and
             list of POI placekeys from safegraph (ALL available options, not sorted)
    """
    df_poi = pd.read_csv(FILE)
    results = pd.DataFrame(columns=['longitude', 'latitude', 'radius', 'time'])

    # create the source and search radius within the timeframe
    all_lons, all_lats, all_rad, all_times = [], [], [], []
    for t in range(dur):
        num_queries = Q

        # list of longitudes and latitudes
        lons = np.random.uniform(bounds_2[1], bounds_1[1], num_queries)
        lats = np.random.uniform(bounds_1[0], bounds_2[0], num_queries)
        rads = [radius]*num_queries
        time = [t]*num_queries

        all_lons.append(lons)
        all_lats.append(lats)
        all_rad.append(rads)
        all_times.append(time)

    results['longitude'] = list(itertools.chain(*all_lons))
    results['latitude'] = list(itertools.chain(*all_lats))
    results['radius'] = list(itertools.chain(*all_rad))
    results['time'] = list(itertools.chain(*all_times))

    destinations = []
    dists = []
    correct = []

    for index, query in results.iterrows():
        df_poi['haversine_distance'] = haversine_dist(source=[query['latitude'], query['longitude']],
                                                      latitude=df_poi['latitude'],
                                                      longitude=df_poi['longitude'])

        dests = df_poi[df_poi['haversine_distance'] <= query['radius']]

        destinations.append(dests['placekey'].tolist())
        dists.append(dests['haversine_distance'].tolist())
        correct.append(len(dests['haversine_distance'].tolist()) > 0)

    results['pois'] = destinations
    results['dists'] = dists
    results['correct'] = correct

    # remove empty queries
    results.drop(results.loc[results['correct'] == False].index, inplace=True)
    results.drop(['correct'], axis=1, inplace=True)

    print(len(results))

    if os.path.exists(SAVE_TO):
        results.to_csv(SAVE_TO, index=False, mode='a', header=False)
    else:
        results.to_csv(SAVE_TO, index=False)


def generate(Q, dur, radius, SAVE_TO, bounds, FILE):
    get_queries(bounds_1=bounds[0],
                bounds_2=bounds[1],
                Q=int(Q),
                dur=dur,
                radius=radius,
                SAVE_TO=SAVE_TO,
                FILE=FILE)