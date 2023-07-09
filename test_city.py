import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def POI_layout(SAVE_TO):
    df = pd.read_csv(SAVE_TO)
    df['latitude'] = df.latitude.apply(float)
    df['longitude'] = df.longitude.apply(float)
    lats = np.array(df['latitude'])
    lons = np.array(df['longitude'])

    plt.scatter(lons, lats, c='red',alpha=0.1, s=400, linewidths=0)
    plt.scatter(lons, lats, c='red',alpha=1, s=5)
    plt.savefig('files/img/test_city/POI_layout.png', bbox_inches='tight')
    plt.show()


def POI_query_layout(Q_RES, POIS):
    df = pd.read_csv(Q_RES)
    pois = pd.read_csv(POIS)

    df['latitude'] = df.latitude.apply(float)
    df['longitude'] = df.longitude.apply(float)
    lats = np.array(df['latitude'])
    lons = np.array(df['longitude'])

    pois['latitude'] = pois.latitude.apply(float)
    pois['longitude'] = pois.longitude.apply(float)
    lats_pois = np.array(pois['latitude'])
    lons_pois = np.array(pois['longitude'])

    plt.scatter(lons, lats, c='red',alpha=0.1, s=400, linewidths=0)
    plt.scatter(lons, lats, c='red',alpha=1, s=5, label='POIs')
    plt.scatter(lons_pois, lats_pois, c='blue', s=10, marker='^', label='queries')
    plt.legend()
    plt.savefig('files/img/test_city/POI_queries_layout.png', bbox_inches='tight')
    plt.show()


def createPOIs(bounds_1, bounds_2, SAVE_TO, SIZE):
    total_num_queries = SIZE

    #  is random
    lons = np.random.uniform(bounds_2[1], bounds_1[1], int(total_num_queries))
    lats = np.random.uniform(bounds_1[0], bounds_2[0], int(total_num_queries))

    areas = np.random.choice([x for x in range(50,1000)], size=total_num_queries)
    placekey = [x for x in range(total_num_queries)]
    city = ['test_city']*total_num_queries
    categoty = ['A']*total_num_queries
    avg_dwell_time = np.random.choice([x for x in range(15,240)], size=total_num_queries) # up to 4 hours
    # only create expected visits for 4 hours (# visitors/hour)
    visits = [np.random.choice([x for x in range(1,201)], size=9).tolist() for y in range(total_num_queries)]

    df = pd.DataFrame(columns=['longitude', 'latitude', 'placekey', 'city', 'geodesic_area',
                               'top_category', 'avg_visits', 'avg_dwell_time'])
    df['longitude'] = lons
    df['latitude'] = lats
    df['placekey'] = placekey
    df['city'] = city
    df['geodesic_area'] = areas
    df['top_category'] = categoty
    df['avg_visits'] = visits
    df['avg_dwell_time'] = avg_dwell_time

    df.to_csv(SAVE_TO, index=False)
