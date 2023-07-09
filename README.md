# RiskAwarePOIRecommender

We provide below a step-by-step set of examples in order to use the recommendation models.

We can not share the real-world mobility data from SafeGraph. 
You can gain accesss to *Patterns*, *Geometry* and *Places* datasets, from 
[Advan](https://advanresearch.com) and [SafeGraph](https://www.safegraph.com).  

In the following set of experiments, we work on synthetic POIs over an unoccupied area in order to redirect the 
focus of our experiments to the performance our the models. As such, we came up with an anonymous city named 
*test_city*. 

#### Step 1: generate test_city's layout

`setup.ipynb`: generates test_city's POI layout, user queries, and outputs figures under *files/img/test_city*.

* we randomly arranged 1,000 POIs 
* we simulate 100 random queries per second for a total of 2,000 queries (20 seconds).
* these parameters can be updated in *setup.ipynb*

Below is a figure for a working example for test_city:
![test_city POI and query layout](files/img/test_city/POI_queries_layout.png)

#### Step 2: go over working examples focusing on model's usibility

`example.ipynb`: provides various use cases and demonstrations on how to use the different recommendation models.
More specifically,
* it shows how to set K, invocation frequecy, and duration of experiment
* it currently supports invocation frequencies of 1, 1/2, 1/4, 1/8 and 1/10 Hz. However, these values can be manually 
  changed in `setup.ipynb` under *rearrange queries to represent different invocation frequencies*
* results can be saved by calling *rec.save(SAVE_TO)*
* get the updated POI occupancies results by calling *rec.get_POIs()['avg_visits']
* get the recommendations for each user by calling *rec.get_top_k()*