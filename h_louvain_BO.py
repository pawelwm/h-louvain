#import csv
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import adjusted_mutual_info_score as AMI 
import hypernetx as hnx
import hypernetx.algorithms.hypergraph_modularity as hmod
#from collections import Counter
#import igraph as ig
import pandas as pd
#import numpy as np
import csv

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

#from collections import Counter
#from scipy.stats import binom 
#from statistics import mean 
#import seaborn as sns
#import matplotlib.pyplot as plt

from h_louvain import hLouvain, last_step, load_ABCDH_from_file
from h_louvain import last_step




class hLouvainBO(hLouvain):
    
    def set_params(
        self,
        seeds = [1234,5325,5467,4723,999,989,1245, 432,1904,7633],#,1234,5325,5467,4723,999],
        xi =1e-3,
        init_points=5,
        n_iter=5,
        pbounds = {'b': (0,1), 'c': (0.01,0.99)},
        bomode = "last_step",
        last_step_top_points = 1,
        hmod_type = hmod.linear,
        show_bo_table = True,
        weights = "standard",
        custom_weights_array = [],
    ):
        self.seeds = seeds
        self.xi = xi
        self.init_points = init_points
        self.pbounds = pbounds
        self.n_iter = n_iter
        self.bomode = bomode
        self.hmod_type = hmod_type
        self.show_bo_table = show_bo_table
        self.last_step_top_points = last_step_top_points
        #self.given_points = [{"b": 0.8, "c": 0.2}, {"b": 0.9, "c": 0.3}]
        self.given_points = []
        self.dts.clear()
        if weights == "custom":
            self.hmod_weights = custom_weights_array
            self.hmod_type = self.custom_weights

        
        self.final_results = pd.DataFrame(
            columns=[
                "b",
                "c",
                "seed",
                "#com",
                "qH",
                "alphas",
                "A",
            ]
        )
     
    
    def custom_weights(self,d, c):

        return self.hmod_weights[c,d]
        
    
    def target_function(self,b,c):
        alphas = []
        qHs =[]
        for i in range(30):
            alphas.append(1-((1-b)**i))
        seeds = self.seeds
        for seed in seeds:
            if self.bomode == "basic" or self.bomode == "last_step":
                A, q2, alphas_out = self.h_louvain_community(alphas = alphas,
                                                    change_mode="communities", 
                                                       change_frequency=c, random_seed=seed)
            if self.bomode == "last_step_all":
                A, q2, alphas_out = self.h_louvain_community_plus_last_step(alphas = alphas,
                                                    change_mode="communities", 
                                                       change_frequency=c, random_seed=seed)
                
                
            qH = self.combined_modularity(A, self.hmod_type, 1)
            
            self.final_results = pd.concat(
                                [
                                    self.final_results,
                                    pd.DataFrame(
                                        [
                                            [b,
                                             c,
                                             seed,
                                             len(A),
                                             qH,
                                             alphas_out,
                                             A,                                                
                                            ]
                                        ],
                                        columns=self.final_results.columns,
                                    ),
                                ],
                                ignore_index=True,
                            )
            
            qHs.append(qH)
         #   print("gH =",qH)
        result = sum(qHs)/len(seeds)
      #  print("av",result)
       # print("c",c,"b",b,"avH =",result)
        return(result)

    def hLouvain_perform_BO(self):
        optimizer = BayesianOptimization(
            f=self.target_function,
            pbounds=self.pbounds,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=self.random_seed,
        )
        if self.show_bo_table == False:
            optimizer._verbose = 0

        acquisition_function = UtilityFunction(kind="ei", xi=self.xi)
    
        for point in self.given_points:
            optimizer.probe(
                params=point,
                lazy=True,
            )
            
        optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
            acquisition_function=acquisition_function
        )
        
        if self.bomode == "last_step":
            to_check = self.final_results.sort_values(by = ["qH"],  ascending=False).head(self.last_step_top_points)
            qH2_list = []
            A2_list = []
            for index, row in to_check.iterrows():
                A2_item = last_step(self.HG, row["A"], self.hmod_type)
                qH2_item = self.combined_modularity(A2_item, self.hmod_type, 1)
                A2_list.append(A2_item)
                qH2_list.append(qH2_item)
            to_check = to_check.assign(A_lstep=A2_list)
            to_check = to_check.assign(qH_lstep=qH2_list)
        
            return to_check
        else:
            return self.final_results
            
            



def main():


    HG = load_ABCDH_from_file("datasets/hg-strict_he.txt")


    hBO = hLouvainBO(HG,hmod_type=hmod.strict, delta_it = 0.0001, delta_phase = 0.0001, random_seed = 875)
    #hBO.hmod_type = hmod.strict
    hBO.set_params(bomode="last_step", hmod_type = hmod.strict, show_bo_table=False)
    hBO_df_results = hBO.hLouvain_perform_BO()

    print(hBO_df_results['qH_lstep'])



if __name__ == "__main__":
    main()
