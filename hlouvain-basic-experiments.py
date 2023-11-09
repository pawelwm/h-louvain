from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import csv
import random
import sys
from copy import deepcopy

import igraph as ig      ## pip install python-igraph
import hypernetx as hnx
import hypernetx.algorithms.hypergraph_modularity as hmod
import hypernetx.algorithms.generative_models as gm
import pickle
from collections import defaultdict
#from h_louvain_decomposed import combined_louvain_constant_alpha, combined_modularity, combined_louvain_alphas
from h_louvain import hLouvain
from h_louvain import h_modularity


from sklearn.metrics import adjusted_rand_score as ARI #ARI(pred, ground) (symmetric true <->pred)
from sklearn.metrics import adjusted_mutual_info_score as AMI #AMI(labels_true, labels_pred, *, average_method='arithmetic') (symmetric true <->pred)

class base_experiment:
    def __init__(
        self,
        search_methods: dict,
        savefile: str,
        abcd_input: str,
        gt_abcd_input: str,
        dataset_short_name: str,
    #    random_seed: int,
        delta_iteration: float,
        delta_phase: float,
        change_mode: str, 
        after_changes: int,
        community_factor: int,
        verbosity: bool = False,
        details: bool = False,
        ground: bool = False,
    ) -> None:
        self.search_methods = search_methods
        self.savefile = savefile
        self.abcd_input = abcd_input
        self.gt_abcd_input = gt_abcd_input
        self.dataset_short_name = dataset_short_name
        self.verbosity = verbosity
        self.details = details
        self.ground = ground
    #    self.random_seed = random_seed
        self.delta_iteration = delta_iteration
        self.delta_phase = delta_phase
        self.change_mode = change_mode 
        self.after_changes = after_changes
        self.community_factor = community_factor


class Grid_search_hclustering(base_experiment):
    def __init__(self, config_filepath: str) -> None:
        self._load_config(config_filepath)
        super().__init__(
            search_methods=self.config["search_methods"],
            savefile=Path(__file__).parent / "results_df" / self.config["savefile"],
        #    random_seed=self.config["random_seed"],
            delta_iteration=self.config["delta_iteration"],
            dataset_short_name=self.config["dataset_short_name"],
            abcd_input=self.config["abcd_input"],
            gt_abcd_input=self.config["gt_abcd_input"],
            delta_phase=self.config["delta_phase"],
            verbosity=self.config["verbosity"],
            ground=self.config["ground"],
            details=self.config["details"],
            change_mode=self.config["change_mode"],
            after_changes=self.config["after_changes"],
            community_factor=self.config["community_factor"],
        )
        if self.config["hmod_type"] == "strict":
            self.hmod_type = hmod.strict
        elif self.config["hmod_type"] == "majority":
            self.hmod_type = hmod.majority
        elif self.config["hmod_type"] == "linear":
            self.hmod_type = hmod.linear
        else: self.hmod_type = hmod.linear  # works as default


    def _load_config(self, config_filepath: str):
        filepath = Path(__file__).parent / "configs" / config_filepath
        with open(filepath, "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)


    def _load_GoT(self):
        ## load the GoT dataset
        Edges, Names, Weights = pickle.load(open( "../../hypernetx/utils/toys/GoT.pkl", "rb" ))
        print(len(Names),'nodes and',len(Edges),'edges')

        ## Nodes are represented as strings from '0' to 'n-1'
        HG = hnx.Hypergraph(dict(enumerate(Edges)))
        ## add edge weights
        for e in HG.edges:
            HG.edges[e].weight = Weights[e]
        ## add full names
        for v in HG.nodes:
            HG.nodes[v].name = Names[v]

        ## compute node strength (add unit weight if unweighted), d-degrees, binomial coefficients
        HG = hmod.precompute_attributes(HG)
        ## build 2-section
        ## G = hmod.two_section(HG)
        return HG

 

    def _load_ABCDH_from_file(self,filename):
        with open(filename,"r") as f:
            rd = csv.reader(f)
            lines = list(rd)

        Edges = []
        for line in lines:
            Edges.append(list(line))

        HG = hnx.Hypergraph(dict(enumerate(Edges)), edge_properties = 'dict', node_properties = 'dict',  properties = 'dict')

        print("HG created")
        print("edges:", len(Edges))
     

        return HG


    def run_experiment(self):

        df_out = pd.DataFrame(
            columns=[
                "dataset",
                "random_seed",
                "method",
                "alphas",
                "phases",
                "communities",
                "changes",
                "iterations",
                "oc-phases",
                "oc-communties",
                "oc-changes",
                "oc-iterations",
                "det-hmod",
                "det-2s",
                "det-opt-fun",
                "det-communities",
                "det-changes",
                "det-phases",
                "h-modularity-type-maximized",
                "combined-modularity-optimized",
                "2s-modularity",
                "strict-h-modularity",
                "majority-h-modularity",
                "linear-h-modularity",
                "gt-strict-h-modularity",
                "gt-majority-h-modularity",
                "gt-linear-h-modularity",
                "ARI",
                "AMI",
                "b",
                "c",
                "A",
            ]
        )
        ds_no = 0
        bi = 0
        ci = 0

        for dataset_file in self.abcd_input:
            HG = self._load_ABCDH_from_file("datasets/"+dataset_file)
            dataset_short_name = self.dataset_short_name[ds_no]
            
            seed_no = 0
            print("Dataset no", ds_no+1,"/",len(self.abcd_input))
            gt_strict = -1
            gt_majority = -1
            gt_linear = -1
            if self.ground == True:
                with open("datasets/"+self.gt_abcd_input[ds_no], 'r') as file:
                    gt = [int(line) for line in file]
                A_gt = [x for x in hmod.dict2part({str(i+1):gt[i] for i in range(len(gt))}) if len(x)>0]
                gt_strict = h_modularity(HG,A_gt,wdc = hmod.strict)
                gt_majority = h_modularity(HG,A_gt,wdc = hmod.majority)
                gt_linear = h_modularity(HG,A_gt,wdc = hmod.linear)

            ds_no+=1
            for seed in self.config["random_seeds"]:
                seed_no+=1
                print("Random experiment", seed_no, "/",len(self.config["random_seeds"]))
                
                for methods in self.search_methods:
                    
                    if methods["method"] == 'constant_level':
                        number_of_levels = methods["number_of_levels"]
                        alphas = []
                        
                        for i in range(number_of_levels):
                            alphas.append(i/max(1,number_of_levels-1))
                        
                        hL = hLouvain(HG,hmod_type=self.hmod_type, 
                                    delta_it = self.delta_iteration, 
                                    delta_phase = self.delta_phase, 
                                    random_seed = seed, details = self.details) 

                        for i in range(number_of_levels):
                            print("alpha = ", alphas[i])
                            A, q2, alphas_out = hL.h_louvain_community(alphas = [alphas[i]])

                            alphas_show = [round(alpha,2) for alpha in alphas_out]
                            
                            ari=-1
                            ami=-1
                            if self.ground == True:
                                d = hmod.part2dict(A)
                                A4ari = [d[str(i+1)] for i in range(len(HG.nodes))]
                                ari = ARI(gt, A4ari) 
                                ami = AMI(gt, A4ari)
            
        
                            df_out = pd.concat(
                                [
                                    df_out,
                                    pd.DataFrame(
                                        [
                                            [   
                                                dataset_short_name,
                                                seed,
                                                methods["method"],
                                                alphas_show,
                                                str(hL.get_phase_history()),
                                                str(hL.get_communities_history()),
                                                str(hL.get_changes_history()),
                                                str(hL.get_iteration_history()),
                                                str(hL.get_oc_phase_history()),
                                                str(hL.get_oc_communities_history()),
                                                str(hL.get_oc_changes_history()),
                                                str(hL.get_oc_iteration_history()),
                                                str(hL.get_detailed_history_hmod()),
                                                str(hL.get_detailed_history_2s()),
                                                str(hL.get_detailed_history_opt_fun()),
                                                str(hL.get_detailed_history_communities()),
                                                str(hL.get_detailed_history_changes()),
                                                str(hL.get_detailed_history_phases()),
                                                self.config["hmod_type"],
                                                q2,
                                                hL.combined_modularity(A, self.hmod_type, 0),
                                                hL.combined_modularity(A, hmod.strict, 1),
                                                hL.combined_modularity(A, hmod.majority, 1),
                                                hL.combined_modularity(A, hmod.linear, 1),
                                                gt_strict,
                                                gt_majority,
                                                gt_linear,
                                                ari,
                                                ami,
                                                0,
                                                0,
                                                A,
                                            ]
                                        ],
                                        columns=df_out.columns,
                                    ),
                                ],
                                ignore_index=True,
                            )

                    if methods["method"] == 'given_alphas':
                        hL = hLouvain(HG,hmod_type=self.hmod_type, 
                                    delta_it = self.delta_iteration, 
                                    delta_phase = self.delta_phase, 
                                    random_seed = seed, details = self.details)  

                        for alphas in methods["alphas"]:
                            print("alphas:", alphas)
                        
                            A, q2, alphas_out = hL.h_louvain_community(alphas = alphas, 
                                        change_mode=self.change_mode, change_frequency=self.community_factor, after_changes=self.after_changes)

                            alphas_show = [round(alpha,2) for alpha in alphas_out]

                            ari=-1
                            ami=-1
                            if self.ground == True:
                                d = hmod.part2dict(A)
                                A4ari = [d[str(i+1)] for i in range(len(HG.nodes))]
                                ari = ARI(gt, A4ari) 
                                ami = AMI(gt, A4ari)
                
            
                            df_out = pd.concat(
                                [
                                    df_out,
                                    pd.DataFrame(
                                        [
                                            [   
                                                    #"ABCD750",
                                                    dataset_short_name,
                                                    # "ChungLu750",
                                                    seed,
                                                    methods["method"],
                                                    alphas_show,
                                                    str(hL.get_phase_history()),
                                                    str(hL.get_communities_history()),
                                                    str(hL.get_changes_history()),
                                                    str(hL.get_iteration_history()),
                                                    str(hL.get_oc_phase_history()),
                                                    str(hL.get_oc_communities_history()),
                                                    str(hL.get_oc_changes_history()),
                                                    str(hL.get_oc_iteration_history()),
                                                    str(hL.get_detailed_history_hmod()),
                                                    str(hL.get_detailed_history_2s()),
                                                    str(hL.get_detailed_history_opt_fun()),
                                                    str(hL.get_detailed_history_communities()),
                                                    str(hL.get_detailed_history_changes()),
                                                    str(hL.get_detailed_history_phases()),
                                                    self.config["hmod_type"],
                                                    q2,
                                                    hL.combined_modularity(A, self.hmod_type, 0),
                                                    hL.combined_modularity(A, hmod.strict, 1),
                                                    hL.combined_modularity(A, hmod.majority, 1),
                                                    hL.combined_modularity(A, hmod.linear, 1),
                                                    gt_strict,
                                                    gt_majority,
                                                    gt_linear,
                                                    ari,
                                                    ami,
                                                    0,
                                                    0,
                                                    A,
                                            ]
                                        ],
                                        columns=df_out.columns,
                                    ),
                                ],
                                ignore_index=True,
                            )

                    if methods["method"] == 'grid-bc':
                        hL = hLouvain(HG,hmod_type=self.hmod_type, 
                                    delta_it = self.delta_iteration, 
                                    delta_phase = self.delta_phase, 
                                    random_seed = seed, details = self.details)  

                        b = [0,0.05,0.1,0.3,0.5,0.7,0.9,0.95,1]
                        c = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

                        all_alphas = []
                        k=0
                        d = methods["d_par"]
                        for bi in b:
                            all_alphas.append([])
                            for i in range(30):
                             #   if bi == 0 and i == 0:
                             #       all_alphas[k].append(1)
                             #   else:
                                all_alphas[k].append(1-((1-bi)**i)*(1-d))
                        
                            k+=1

                        #print(all_alphas)
                        
                        for ci in c:
                            k=0
                            for alphas in all_alphas:
                                k+=1
                                print("b:", b[k-1], "c:", ci)
                            
                                A, q2, alphas_out = hL.h_louvain_community(alphas = alphas, 
                                            change_mode=self.change_mode, change_frequency=ci, after_changes=self.after_changes)

                                alphas_show = [round(alpha,4) for alpha in alphas_out]

                                print("alphas:", alphas_show)

                                ari=-1
                                ami=-1
                                if self.ground == True:
                                    d = hmod.part2dict(A)
                                    A4ari = [d[str(i+1)] for i in range(len(HG.nodes))]
                                    ari = ARI(gt, A4ari) 
                                    ami = AMI(gt, A4ari)
                    
                
                                df_out = pd.concat(
                                    [
                                        df_out,
                                        pd.DataFrame(
                                            [
                                                [   
                                                        #"ABCD750",
                                                        dataset_short_name,
                                                        # "ChungLu750",
                                                        seed,
                                                        methods["method"],
                                                        alphas_show,
                                                        str(hL.get_phase_history()),
                                                        str(hL.get_communities_history()),
                                                        str(hL.get_changes_history()),
                                                        str(hL.get_iteration_history()),
                                                        str(hL.get_oc_phase_history()),
                                                        str(hL.get_oc_communities_history()),
                                                        str(hL.get_oc_changes_history()),
                                                        str(hL.get_oc_iteration_history()),
                                                        str(hL.get_detailed_history_hmod()),
                                                        str(hL.get_detailed_history_2s()),
                                                        str(hL.get_detailed_history_opt_fun()),
                                                        str(hL.get_detailed_history_communities()),
                                                        str(hL.get_detailed_history_changes()),
                                                        str(hL.get_detailed_history_phases()),
                                                        self.config["hmod_type"],
                                                        q2,
                                                        hL.combined_modularity(A, self.hmod_type, 0),
                                                        hL.combined_modularity(A, hmod.strict, 1),
                                                        hL.combined_modularity(A, hmod.majority, 1),
                                                        hL.combined_modularity(A, hmod.linear, 1),
                                                        gt_strict,
                                                        gt_majority,
                                                        gt_linear,
                                                        ari,
                                                        ami,
                                                        b[k-1],
                                                        ci,
                                                        A,
                                                ]
                                            ],
                                            columns=df_out.columns,
                                        ),
                                    ],
                                    ignore_index=True,
                                )
        self.save(df_out)

    def load_datasets(self):
        return [5]


    def save(self, df):
        Path(self.savefile).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(self.savefile))

    def _procedureA(
        self,
        df
    ):
         return df



def main():
    gsh = Grid_search_hclustering(config_filepath=sys.argv[1])
    gsh.run_experiment()


if __name__ == "__main__":
    main()
