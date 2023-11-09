from distutils.util import change_root
import pandas as pd
import numpy as np
import igraph as ig      ## pip install python-igraph
import hypernetx as hnx
import hypernetx.algorithms.hypergraph_modularity as hmod
import hypernetx.algorithms.generative_models as gm
import pickle
from collections import defaultdict
import copy
import math
import csv
from collections import Counter
from scipy.stats import binom 
from scipy.special import comb


#precalculation of d_weights (for purposes of degree tax change calculation)

def d_weights(H):

    ## all same edge weights?
    uniq = (len(Counter(H.edges.properties['weight']))==1)
    
  
    if uniq:        
        Ctr = Counter([H.size(i) for i in H.edges])

    else:
        m = np.max([H.size(i) for i in H.edges])
        Ctr2 = np.repeat(0,1+m)
        for e in H.edges:
            w = H.edges[e].weight
            Ctr2[H.size(e)] += w  

        Ctr = Counter([len(H.edges[e]) for e in H.edges])
        for k in range(len(Ctr2)):
            Ctr[k] = Ctr2[k]
    
    
    # 3. compute binomial coeffcients (modularity speed-up)
    bin_coef = {}
    
    for n in Ctr.keys():
        for k in np.arange(n // 2 + 1, n + 1):
            bin_coef[(n, k)] = comb(n, k, exact=True)
    
    return Ctr, bin_coef


#optimized version of modularity calculation for hnx2.0 (credits to F.Theberge)

def h_modularity(H, A, wdc=hmod.linear):

    ## all same edge weights?
    uniq = (len(Counter(H.edges.properties['weight']))==1)
    
    ## Edge Contribution
    H_id = H.incidence_dict
    d = hmod.part2dict(A)
    L = [ [d[i] for i in H_id[x]] for x in H_id ]
    

    ## all same weight
    if uniq:
        _ctr = Counter([ (Counter(l).most_common(1)[0][1],len(l)) for l in L])
        EC = sum([wdc(k[1],k[0])*_ctr[k] for k in _ctr.keys() if k[0] > k[1]/2])
    else:
        _keys = [ (Counter(l).most_common(1)[0][1],len(l)) for l in L]
        _vals = list(H.edge_props['weight']) ## Thanks Brenda!!
        _df = pd.DataFrame(zip(_keys,_vals), columns=['key','val'])
        _df = _df.groupby(by='key').sum()
        EC = sum([ wdc(k[1],k[0])*v[0] for (k,v) in _df.iterrows() if k[0]>k[1]/2 ])
        
    ## Degree Tax
    if uniq:        
        VolA = [sum([H.degree(i) for i in k]) for k in A]
        Ctr = Counter([H.size(i) for i in H.edges])

    else:
        ## this is the bottleneck
        VolA = np.repeat(0,1+np.max(list(d.values())))
        m = np.max([H.size(i) for i in H.edges])
        Ctr = np.repeat(0,1+m)
        S = 0
        for e in H.edges:
            w = H.edges[e].weight
            Ctr[H.size(e)] += w  
            S += w
            for v in H.edges[e]:
                VolA[d[v]] += w 
                
    VolV = np.sum(VolA)
    VolA = [x/VolV for x in VolA]
    DT = 0
    
    if uniq:        
        for d in Ctr.keys():
            Cnt = Ctr[d]
            for c in np.arange(int(np.floor(d/2+1)),d+1):
                for Vol in VolA:
                    DT += (Cnt*wdc(d,c)*binom.pmf(c,d,Vol))
        return (EC-DT)/H.number_of_edges()
    else:
        for d in range(len(Ctr)):
            Cnt = Ctr[d]
            for c in np.arange(int(np.floor(d/2+1)),d+1):
                for Vol in VolA:
                    DT += (Cnt*wdc(d,c)*binom.pmf(c,d,Vol))
        return (EC-DT)/S

# calculation of exponential part of binomial
def bin_ppmf(d, c, p):
    return p**c * (1 - p)**(d - c)

# degree tax for node subset of given total volume (sum of nodes' weights)


class hLouvain:
    def __init__(
        self,
        HG,
        hmod_type = hmod.linear,
        delta_it = 0.00001, 
        delta_phase = 0.00001, 
        random_seed = 123,
        details = False,
    ) -> None:
        self.HG = HG
        self.hmod_type = hmod_type
        self.delta_it = delta_it
        self.delta_phase = delta_phase
        self.random_seed = random_seed
        self.details = details
        self.G = hmod.two_section(HG)
        self.h_nodes = []
        for e in self.HG.E:
            E = self.HG.E[e]
            for node in E:
                if node not in self.h_nodes:
                    self.h_nodes.append(node)
    
        self.startHGdict, self.startA = self.build_HG_dict_from_HG()
        self.HGdict = {},
        self.dts = defaultdict(float) #dictionary containing degree taxes for given volumes
   
        self.changes = 0
        self.communities = len(self.startA)
        self.phase = 1
        self.iteration = 1
        self.change_mode = "iter"
        self.d_weights, self.bin_coef = d_weights(self.HG)
        self.total_weight = sum(self.d_weights.values())
        # additional logging
        self.phase_history = []
        self.community_history = []
        self.changes_history = []
        self.iteration_history = []
        self.onchange_phase_history = []
        self.onchange_community_history = []
        self.onchange_changes_history = []
        self.onchange_iteration_history = []
        self.detailed_history_phases = []
        self.detailed_history_hmod = []
        self.detailed_history_2s = []
        self.detailed_history_communities = []
        self.detailed_history_changes = []
        self.detailed_history_opt_fun = []


    def get_detailed_history_phases(self):
        return self.detailed_history_phases

    def get_detailed_history_hmod(self):
        return self.detailed_history_hmod

    def get_detailed_history_2s(self):
        return self.detailed_history_2s

    def get_detailed_history_communities(self):
        return self.detailed_history_communities

    def get_detailed_history_changes(self):
        return self.detailed_history_changes

    def get_detailed_history_opt_fun(self):
        return self.detailed_history_opt_fun

    def get_phase_history(self):
        return self.phase_history

    def get_changes_history(self):
        return self.changes_history

    def get_communities_history(self):
        return self.community_history

    def get_iteration_history(self):
        return self.iteration_history

    def get_oc_phase_history(self):
        return self.onchange_phase_history

    def get_oc_changes_history(self):
        return self.onchange_changes_history

    def get_oc_communities_history(self):
        return self.onchange_community_history

    def get_oc_iteration_history(self):
        return self.onchange_iteration_history

    def _setHGDictAndA(self):
        self.HGdict = copy.deepcopy(self.startHGdict)
        A = copy.deepcopy(self.startA)
        return A
    
    
    def _hg_neigh_dict(self):
        """
        Optimizing the access to nodes neighbours by additional dictionary based on hnx neighbours() function

        Parameters
        HG : an HNX hypergraph

        Returns
        -------
        : dict
        a dictionary with {node: list of neighboring nodes}
        """

        result = {}
        for v in self.h_nodes:
            result[v] = self.HG.neighbors(v)
        return result


    def combined_modularity(self,A, hmod_type=hmod.linear, alpha=0.5):
        
        # Exclude empty clusters if exist
        A = [a for a in A if len(a) > 0]
        
        # Calculation of hypergraph modularity (based on hypernetx)
        hyper = h_modularity(self.HG,A, wdc=hmod_type) # we use linear (default)  - other predifined possibilities are called majority/strict)
        
        # Calculation of modularity of 2-section graph (based on iGraph)
        d = hmod.part2dict(A)
        partition  = [d[i] for i in self.h_nodes]
        twosect = self.G.modularity(partition,weights='weight')  # weighting is enabled
        
        return (1-alpha)*twosect+alpha*hyper




    def _ec_loss(self, c, s, wdc=hmod.linear):
        
        ec_h = 0  # ec for hypergraph
        ec_2s = 0  #ec for 2section graph
        
        #Edge-by-edge accumulation of edge contribution loss (for hypergraph and 2section)
        for e in self.HGdict["v"][s]["e"].keys():
            d = self.HGdict["e"][e]["size"]
            if d > 1:  #we ignore hyperedges with one vertex
                w = self.HGdict["e"][e]["weight"]
                old_c = self.HGdict["c"][c]["e"][e] # counting vertices of edge e in community c
                new_c = old_c - self.HGdict["v"][s]["e"][e] #updating after taking the supernode     
                ec_h += w * (wdc(d, old_c)- wdc(d, new_c))
                ec_2s += w * self.HGdict["v"][s]["e"][e] * new_c/(d-1)
        return ec_h / self.total_weight, 2*ec_2s / self.total_volume 

    # calculating edge contribution gain when joinig supernode s to destination community c

    def _ec_gain(self, c, s, wdc=hmod.linear):
        

        ec_h = 0
        ec_2s = 0
        
        #Edge-by-edge accumulation of edge contribution change 
        for e in self.HGdict["v"][s]["e"].keys():
            d = self.HGdict["e"][e]["size"]
            if d > 1:  #we ignore hyperedges with one vertex
                w = self.HGdict["e"][e]["weight"]
                old_c = self.HGdict["c"][c]["e"][e]
                new_c = old_c + self.HGdict["v"][s]["e"][e] 
                ec_h += w * (wdc(d, new_c) - wdc(d, old_c))
                ec_2s += w * self.HGdict["v"][s]["e"][e] * old_c/(d-1)
        
        return ec_h / self.total_weight, 2*ec_2s / self.total_volume 


    def _degree_tax_hyper(self, volume, wdc=hmod.linear):
        
        volume = volume/self.total_volume
        
        DT = 0
        for d in self.d_weights.keys():
            x = 0
            for c in np.arange(int(np.floor(d / 2)) + 1, d + 1):
                x += self.bin_coef[(d, c)] * wdc(d, c) * bin_ppmf(d, c, volume)
            DT += x * self.d_weights[d]
        return DT / self.total_weight

    def _combined_delta(self,delta_h, delta_2s,alpha=0.5):
        return (1-alpha)*delta_2s+alpha*delta_h

    #procedure for building dictionary-based data structures for hypergraph with supernodes
    #procedure produce the initial structure based on the input hypergraph with all supernodes with just only node

    def build_HG_dict_from_HG(self):
        
        HGdict = {}
        HGdict["v"]={}  #subdictionary storing information on vertices
        HGdict["c"]={}  #subdictionary storing information on current communities
        A = []          #current partition as a list of sets
        name2index = {} 
        
        # preparing structures for vertices and initial communities
        
        for i in range(len(self.h_nodes)):
            HGdict["v"][i]={}
            HGdict["v"][i]["name"] = [self.h_nodes[i]]
            HGdict["v"][i]["e"] = defaultdict(int) #dictionary  edge: edge_dependent_vertex_weight   
            HGdict["v"][i]["strength"]=0   #strength = weigthed degree'
            HGdict["c"][i]={}
            HGdict["c"][i]["e"] = defaultdict(int) #dictionary  edge: edge_dependent_community_weight
            HGdict["c"][i]["strength"]=0
            A.append({self.h_nodes[i]})
            name2index[self.h_nodes[i]] = i

        # preparing structures for edges 
            
        HGdict["e"] = {}
        edge_dict = copy.deepcopy(self.HG.incidence_dict)
        
        HGdict["total_volume"] = 0    # total volume of all vertices             
        HGdict["total_weight"] = sum(self.HG.edges[j].weight for j in self.HG.edges) #sum of edges weights
        
        for j in range(len(self.HG.edges)):
            HGdict["e"][j] = {}
            HGdict["e"][j]["v"] = defaultdict(int)  #dictionary  vertex: edge_dependent_vertex_weight
            HGdict["e"][j]["weight"] = copy.deepcopy(self.HG.edges[j].weight)
            HGdict["e"][j]["size"] = copy.deepcopy(self.HG.size(j))
            HGdict["e"][j]["volume"] = HGdict["e"][j]["weight"]*HGdict["e"][j]["size"]    
            for t in edge_dict[j]:
                HGdict["e"][j]["v"][name2index[t]] += 1 
                HGdict["v"][name2index[t]]["e"][j] += 1
                HGdict["c"][name2index[t]]["e"][j] += 1
                HGdict["v"][name2index[t]]["strength"] += HGdict["e"][j]["weight"]
                HGdict["c"][name2index[t]]["strength"] += HGdict["e"][j]["weight"]  
                HGdict["total_volume"] += HGdict["e"][j]["weight"]

                
        return HGdict, A


    # procedure for node_collapsing
    # procedure produce supervertices based on communities from the previous phase

    def node_collapsing(self,HGd,A):
        
        # reindexing the partition from previous phase (skipping empty indexes)
        newA = [a for a in A if len(a) > 0]
        
        # building new vertices and communities (by collapsing nodes to supernodes)
        
        HGdict = {}
        HGdict["v"]={}
        HGdict["c"]={}
        i = 0
        for j in range(len(A)):
            if len(A[j])>0:
                HGdict["v"][i]={}
                HGdict["c"][i]={}
                HGdict["v"][i]["e"]={}
                HGdict["c"][i]["e"]=defaultdict(int)
                for e in HGd["c"][j]["e"].keys():
                    if HGd["c"][j]["e"][e] > 0:
                        HGdict["v"][i]["e"][e] = HGd["c"][j]["e"][e]
                        HGdict["c"][i]["e"][e] = HGd["c"][j]["e"][e]
                HGdict["v"][i]["strength"] = HGd["c"][j]["strength"]
                HGdict["c"][i]["strength"] = HGd["c"][j]["strength"]
                i+=1        

        #updating edges based on new supernodes indexes and weights
        
        HGdict["e"] = HGd["e"]
        HGdict["total_volume"] = HGd["total_volume"]
        HGdict["total_weight"] = HGd["total_weight"]
        for j in range(len(HGdict["e"])):
            HGdict["e"][j]["v"] = {}
            HGdict["e"][j]["v"] = defaultdict(int)  
        for v in HGdict["v"].keys():
            for e in HGdict["v"][v]["e"].keys():
                HGdict["e"][e]["v"][v] = HGdict["v"][v]["e"][e] 
                
        return HGdict, newA


    def next_maximization_iteration(self, L, DL, A1, D, 
                alpha = 0.5):
        
        current_alpha = alpha
        A = copy.deepcopy(A1)

        local_length = len(A)
        step = math.ceil(local_length/10)
        k = 0
        
        for sn in list(np.random.RandomState(seed=self.random_seed).permutation(L)): #permute supernodes (L contains initial partition, so supernodes)
            #check the current cluster number (using the first node in supernode)
            c = D[list(sn)[0]] 
            #check the supernode index
            si = DL[list(sn)[0]] 
            
            # matrix with modularity deltas
            
            M = []       
            superneighbors = list(set(np.concatenate([[D[v] for v in self.neighbors_dict[w]] for w in sn])))
            if len(superneighbors) > 1 or c not in superneighbors: #checking only clusters with some neighbors of current supernodes
                
                
                # calculating loss in degree tax for hypergraph when taking vertex si from community c
                vols = self.HGdict["v"][si]["strength"]
                volc = self.HGdict["c"][c]["strength"]

               

                if not self.dts[volc]:
                    self.dts[volc] = self._degree_tax_hyper(volc, wdc=self.hmod_type)
                if volc > vols and not self.dts[volc - vols]:
                    self.dts[volc-vols] = self._degree_tax_hyper(volc - vols, wdc=self.hmod_type)
                
                dt_h_loss = self.dts[volc] - self.dts[volc-vols] 
                
                # calculating loss in edge contribution for hypergraph and 2section when taking vertex si from community c
                
                ec_h_loss, ec_2s_loss = self._ec_loss( c, si, wdc=self.hmod_type)
                                
                for i in superneighbors:   
                    if c == i:
                        M.append(0) 
                    else:
                        # gain in degree tax for hypergraph when joining vertex si to community i
                
                        voli = self.HGdict["c"][i]["strength"]
                   
                   
                        if not self.dts[voli]:
                            self.dts[voli] = self._degree_tax_hyper(voli, wdc=self.hmod_type)
                        if not self.dts[voli + vols]:
                            self.dts[voli + vols] = self._degree_tax_hyper(voli + vols, wdc=self.hmod_type)
                        dt_h_gain = self.dts[voli + vols] - self.dts[voli]
                        
                        # change in degree tax for 2section graph
                        delta_dt_2s =2*vols*(vols+voli-volc)/(self.total_volume**2)
                        
                        # gains in edge contribution when joining si to community i
                        ec_h_gain, ec_2s_gain = self._ec_gain(i, si, wdc=self.hmod_type)
                        
                        #calulating deltas
                        delta_h = ec_h_gain - ec_h_loss - (dt_h_gain - dt_h_loss)
                        delta_2s = ec_2s_gain - ec_2s_loss - delta_dt_2s
                        
                        M.append(self._combined_delta(delta_h, delta_2s, current_alpha))
                # make a move maximizing the gain 
                if max(M) > 0:
                    i = superneighbors[np.argmax(M)]
                    for v in list(sn):
                        A[c] = A[c] - {v}
                        A[i] = A[i].union({v})
                        D[v] = i

                    self.changes+=1
                    self.new_changes+=1
                    if len(A[c]) == 0:
                        self.communities-=1


                    if self.change_mode == "modifications":
                        if self.new_changes >= self.after_changes:
                            self.level+=1
                            self.new_changes = 0
                            if self.level > len(self.alphas) - 1:
                                self.alphas.append(self.alphas[-1])
                            current_alpha = self.alphas[self.level]
                            self.onchange_phase_history.append(self.phase)
                            self.onchange_changes_history.append(self.changes)
                            self.onchange_community_history.append(self.communities)
                            self.onchange_iteration_history.append(self.iteration)


                    if self.change_mode == "communities":
                        if self.communities <= self.community_threshold:
                            self.level+=1
                            self.community_threshold = self.community_threshold/self.community_factor
                            if self.level > len(self.alphas) - 1:
                                self.alphas.append(self.alphas[-1])
                            current_alpha = self.alphas[self.level]

                            self.onchange_phase_history.append(self.phase)
                            self.onchange_changes_history.append(self.changes)
                            self.onchange_community_history.append(self.communities)
                            self.onchange_iteration_history.append(self.iteration)

                                        
                    for e in self.HGdict["v"][si]["e"].keys():
                        self.HGdict["c"][c]["e"][e] -= self.HGdict["v"][si]["e"][e]
                        self.HGdict["c"][i]["e"][e] += self.HGdict["v"][si]["e"][e]
                        self.HGdict["c"][c]["strength"] -= self.HGdict["v"][si]["e"][e]*self.HGdict["e"][e]["weight"]
                        self.HGdict["c"][i]["strength"] += self.HGdict["v"][si]["e"][e]*self.HGdict["e"][e]["weight"]   

                if (self.details == True) and (k % step == 0):
                    
                    comb, twosec, hyper = self.combined_modularity2(A, self.hmod_type, current_alpha)
                    self.detailed_history_opt_fun.append(comb)
                    self.detailed_history_2s.append(twosec)
                    self.detailed_history_hmod.append(hyper)
                    self.detailed_history_communities.append(self.communities)
                    self.detailed_history_changes.append(self.changes)
                    self.detailed_history_phases.append(self.phase)
                    print(self.changes, self.communities, k)
                k+=1
                    
        q2 = self.combined_modularity(A, self.hmod_type, current_alpha)

        return q2, A, D
        


    def next_maximization_phase(self, L):

        DL = hmod.part2dict(L)    
        A1 = L[:]  
        D = hmod.part2dict(A1) #current partition as a dictionary (D will change during the phase)
        
        self.total_volume = self.HGdict["total_volume"]

        if self.change_mode == "iter":
            if self.level > len(self.alphas) - 1:
                self.alphas.append(self.alphas[-1])
        qC = self.combined_modularity(A1, self.hmod_type, self.alphas[self.level])
        
        while True:

            q2, A2, D =  self.next_maximization_iteration(L, DL, A1, D, self.alphas[self.level])
            qC = self.combined_modularity(A1, self.hmod_type, self.alphas[self.level]) 


            if (q2 - qC) < self.delta_it:

                self.HGdict, newA = self.node_collapsing(self.HGdict,A2)

                break

            self.phase_history.append(self.phase)
            self.changes_history.append(self.changes)
            self.community_history.append(self.communities)
            self.iteration_history.append(self.iteration)
            self.iteration += 1
            if self.change_mode == "iter":
                self.level+=1
                self.onchange_phase_history.append(self.phase)
                self.onchange_changes_history.append(self.changes)
                self.onchange_community_history.append(self.communities)
                self.onchange_iteration_history.append(self.iteration)

                if self.level > len(self.alphas) - 1:
                    self.alphas.append(self.alphas[-1])
                
            A1 = A2
        
        return newA, qC 




    def h_louvain_community(self,alphas = [1], change_mode = "iter", after_changes = 300, change_frequency = 0.5, random_seed = 1):

        A1 = self._setHGDictAndA()
        if random_seed > 1:
            self.random_seed = random_seed
        self.changes = 0
        self.new_changes = 0
        self.communities = len(A1)
        self.phase = 1
        self.phase_history = []
        self.community_history = []
        self.changes_history = []
        self.iteration_history = []
        self.onchange_phase_history = []
        self.onchange_community_history = []
        self.onchange_changes_history = []
        self.onchange_iteration_history = []
        self.change_mode = change_mode
        self.after_changes = after_changes
        self.alphas = alphas
        self.level = 0
        self.iteration = 1
        self.community_factor = 1/change_frequency
        self.community_threshold = self.communities/self.community_factor

        
        self.neighbors_dict = copy.deepcopy(self._hg_neigh_dict())
        q1 = 0
        while True:
            
            A2, qnew  = self.next_maximization_phase(A1)
            q1 = self.combined_modularity(A1,  self.hmod_type, self.alphas[self.level])
            
            A1 = A2
            if (qnew-q1) < self.delta_phase:
                return A2, qnew, self.alphas[0:self.level+1]
            q1 = qnew
            self.phase+=1
            self.iteration = 1
            if self.change_mode == "phase":
                self.level+=1
                self.onchange_phase_history.append(self.phase)
                self.onchange_changes_history.append(self.changes)
                self.onchange_community_history.append(self.communities)
                self.onchange_iteration_history.append(self.iteration)

                if self.level > len(self.alphas) - 1:
                    self.alphas.append(self.alphas[-1])
            


def load_ABCDH_from_file(filename):
    with open(filename,"r") as f:
        rd = csv.reader(f)
        lines = list(rd)

    print("File loaded")

    Edges = []
    for line in lines:
        Edges.append(list(line))

    HG = hnx.Hypergraph(dict(enumerate(Edges)))

    print("HG created")

    print(len(Edges), "edges")
 
    return HG



def main():


    HG = load_ABCDH_from_file("results_3000_he.txt")


    hL = hLouvain(HG,hmod_type=hmod.linear)

    A, q2, alphas_out = hL.h_louvain_community(alphas = [0.5,1,0.5,0.7,0.8,1], change_mode="communities", change_frequency= 0.5)
        
    print("")
    print("FINAL ANSWER:")
    print("partition:",A)
    print("qC:", q2)
    print("alphas", alphas_out)



if __name__ == "__main__":
    main()
