import math
import graph
import random

import matplotlib.pyplot as plt
import numpy as np

def NonMetricGraph(nodes, lim):
    """
    Generating a nonmetric graph in a file (similar to 'sixnodes')
    nodes - the number of nodes in the graph
    """
    for i in range(nodes):
        for j in range((i+1),nodes):
            if i == 0 and j == 1:
                f = open(f'{nodes}nodes', "w")
            else:
                f = open(f'{nodes}nodes', "a")

            edge = int(random.random()*lim) + 1
            f.write(f' {i} {j} {edge}' + '\n')
            f.close()

def XYnodes(n, Xlim, Ylim):
    """
    Generates a node file similar to 'cities25'
    with the coordinates of n nodes
    """
    for i in range(n):
        node = (random.random()*Xlim, random.random()*Ylim)
        if i == 0:
            f = open(f'cities{n}t', "w")
        else:
            f = open(f'cities{n}t', "a")

        f.write(f' {node[0]} {node[1]}' + '\n')
        f.close()


class Test1(object):
    
    def __init__(self, nodes = 10, metric = True):
        """
        Parameters:
        - nodes - the number of nodes in the generated graphs
        - metric - whether the graphs are metric or not (whether they satisfy the triangle inequality)
        """
        self.nodes = nodes
        self.metric = metric
        if metric == True:
            XYnodes(nodes, 500, 500)
            self.g = graph.Graph(-1, f"cities{nodes}t")
        else:
            NonMetricGraph(nodes, nodes)
            self.g = graph.Graph(nodes, f'{nodes}nodes')

        self.data = {}

    def findTour(self):
        """
        Calculates the tourValue of the algorithms 
        and saves it in self.data and self.dataExt
        """
        self.g.swapHeuristic()
        self.data["SwapHeuristic"] = self.g.tourValue()
        self.g.perm = [i for i in range(self.nodes)]

        self.g.TwoOptHeuristic()
        self.data["TwoOptHeuristic"] = self.g.tourValue()
        self.g.perm = [i for i in range(self.nodes)]

        self.g.Greedy()
        self.data["Greedy"] = self.g.tourValue()
        self.g.perm = [i for i in range(self.nodes)]
        
        self.g.Greedy2()
        self.data["Greedy2"] = self.g.tourValue()
        self.g.perm = [i for i in range(self.nodes)]

        self.g.Greedy2Ext()
        self.data["Greedy2Ext"] = self.g.tourValue()
        self.g.perm = [i for i in range(self.nodes)]

        if self.metric == True:
            self.g.Christfides()
            self.data["Christofides"] = self.g.tourValue()
            self.g.perm = [i for i in range(self.nodes)]
        
        if self.nodes <= 15:
            self.data["DPExt"] = self.g.DPAlgorithmExt()

    def ratio(self):
        """
        Calculates the ratio of each algorithm and the optimal solution (given by DP)
        Only for self.nodes<=15
        """
        if self.nodes >15:
            return

        dev = {}
        dev["SwapHeuristic"] = self.data["SwapHeuristic"]/self.data["DPExt"]
        dev["TwoOptHeuristic"] = self.data["TwoOptHeuristic"]/self.data["DPExt"]
        dev["Greedy"] = self.data["Greedy"]/self.data["DPExt"]
        dev["Greedy2"] = self.data["Greedy2"]/self.data["DPExt"]
        dev["Greedy2Ext"] = self.data["Greedy2Ext"]/self.data["DPExt"]
        if self.metric:
            dev["Christofides"] = self.data["Christofides"]/self.data["DPExt"]

        return dev
        

    def plot(self):
        """
        Plots the result tourValues of the algorithms
        Saves the plot as a .png
        """
        plt.plot(list(self.data.values()))
        plt.gca().set_xticks(np.arange(len(list(self.data.keys()))))
        plt.gca().set_xticklabels(list(self.data.keys()), rotation = 15)
        plt.title(f'Tour Value for a Metric Graph of {self.nodes} nodes')
        fig = plt.gcf()
        plt.show()
        plt.draw()
        fig.savefig(f'{self.nodes} Cities Algorithm Data.png')


class Test2(object):
    """
    Generates a number of graphs
    Calculates the tourValue for different algorithms
    Plots the results
    """
    def __init__(self, nodes = 10, tests = 10, metric = True, extended = False):
        """
        Parameters:
        - nodes - the number of nodes in the generated graphs
        - tests - the number of graphs to be generated
        - metric - whether the graphs are metric or not (whether they satisfy the triangle inequality)
        - extended - if True, swapHeuristic() and TwoOptHeurisitc() are used to further improve the result
        """
        self.nodes = nodes
        self.tests = tests
        self.metric = metric
        self.extended = extended
        self.setUp()
        
    def setUp(self):
        """
        Initialises a dictionary to hold the test data
        Generates the graphs
        """
        self.data = {"Greedy": [], "Greedy2": [], "Greedy2Ext": []}
        if self.extended:
            self.dataExt = {"Greedy": [], "Greedy2": [], "Greedy2Ext": []}
        if self.metric == True:
            self.data["Christofides"] = []
            if self.extended:
                self.dataExt["Christofides"] = []

        if self.nodes <= 15:
            self.data["DPExt"] = []
            if self.extended:
                self.dataExt["DPExt"] = []

        self.graphs = []
        if self.metric:
            for i in range(self.tests):
                XYnodes(self.nodes, 500, 500)
                g = graph.Graph(-1, f"cities{self.nodes}t")
                self.graphs.append(g)
        else:
            for i in range(self.tests):
                NonMetricGraph(self.nodes, self.nodes)
                g = graph.Graph(self.nodes, f'{self.nodes}nodes')
                self.graphs.append(g)

    def findTour(self):
        """
        For each graph calculates the tourValue of the algorithms 
        and saves it in self.data and self.dataExt
        """
        for i in range(self.tests):
            self.graphs[i].Greedy()
            self.data["Greedy"].append(self.graphs[i].tourValue())
            if self.extended:
                self.graphs[i].swapHeuristic()
                self.graphs[i].TwoOptHeuristic()
                self.dataExt["Greedy"].append(self.graphs[i].tourValue())
            self.graphs[i].perm = [j for j in range(self.nodes)]

            self.graphs[i].Greedy2(0)
            self.data["Greedy2"].append(self.graphs[i].tourValue())
            if self.extended:
                self.graphs[i].swapHeuristic()
                self.graphs[i].TwoOptHeuristic()
                self.dataExt["Greedy2"].append(self.graphs[i].tourValue())
            self.graphs[i].perm = [j for j in range(self.nodes)]

            self.graphs[i].Greedy2Ext()
            self.data["Greedy2Ext"].append(self.graphs[i].tourValue())
            if self.extended:
                self.graphs[i].swapHeuristic()
                self.graphs[i].TwoOptHeuristic()
                self.dataExt["Greedy2Ext"].append(self.graphs[i].tourValue())
            self.graphs[i].perm = [j for j in range(self.nodes)]

            if self.metric == True:
                self.graphs[i].Christfides()
                self.data["Christofides"].append(self.graphs[i].tourValue())
                if self.extended:
                    self.graphs[i].swapHeuristic()
                    self.graphs[i].TwoOptHeuristic()
                    self.dataExt["Christofides"].append(self.graphs[i].tourValue())
            self.graphs[i].perm = [j for j in range(self.nodes)]

            if self.nodes <= 15:
                cost = self.graphs[i].DPAlgorithmExt()
                self.data["DPExt"].append(cost)
                if self.extended:
                    self.dataExt["DPExt"].append(cost)
        
    def ratio(self):
        """
        Calculates the mean ratio of each algorithm and the optimal solution (given by DP)
        Only for self.nodes<=15
        Returns a dictionary
        """
        if self.nodes > 15:
            return

        dev = {"Greedy": [], "Greedy2": [], "Greedy2Ext": []}
        if self.metric:
            dev["Christofides"] = []
        # adds the ratio of each graph
        for i in range(self.tests):
            dev["Greedy"].append(round(self.data["Greedy"][i]/self.data["DPExt"][i], 2))
            dev["Greedy2"].append(round(self.data["Greedy2"][i]/self.data["DPExt"][i], 2))
            dev["Greedy2Ext"].append(round(self.data["Greedy2Ext"][i]/self.data["DPExt"][i], 2))
            if self.metric:
                dev["Christofides"].append(round(self.data["Christofides"][i]/self.data["DPExt"][i], 2))

        if self.extended:
            devExt = {"Greedy": [], "Greedy2": [], "Greedy2Ext": []}
            if self.metric:
                devExt["Christofides"] = []
            for i in range(self.tests):
                devExt["Greedy"].append(round(self.dataExt["Greedy"][i]/self.dataExt["DPExt"][i], 2))
                devExt["Greedy2"].append(round(self.dataExt["Greedy2"][i]/self.dataExt["DPExt"][i], 2))
                devExt["Greedy2Ext"].append(round(self.dataExt["Greedy2Ext"][i]/self.dataExt["DPExt"][i], 2))
                if self.metric:
                    devExt["Christofides"].append(round(self.dataExt["Christofides"][i]/self.dataExt["DPExt"][i], 2))

        mean_ratio = {}
        # calculates the mean ratio for each algorithm 
        if self.extended:
            for key in dev.keys():
                mean_ratio[key] = [np.mean(dev[key]), np.mean(devExt[key])]
        else:

            for key in dev.keys():
                mean_ratio[key] = np.mean(dev[key])
        return mean_ratio

    def saveToFile(self, filename):
        for i in range(self.tests):
            if i == 0:
                f = open(filename, "w")
            else:
                f = open(filename, "a")
            f.write(f'#{i} Graph')
            f.write(f' Greedy: {self.data["Greedy"][i]}\n')
            f.write(f' Greedy2: {self.data["Greedy2"][i]}\n')
            f.write(f' Greedy2Ext: {self.data["Greedy2Ext"][i]}\n')
            if self.metric:
                f.write(f' Christofides: {self.data["Christofides"][i]}\n')
            if self.nodes <= 15:
                f.write(f' DPExt: {self.data["DPExt"][i]}\n')
            f.close()

    def plot(self):
        """
        Plots the result tourValues of the algorithms
        Saves the plot as a .png
        """
        for key in self.data.keys():
            plt.plot(self.data[key], label = key)
        plt.title(f'{self.nodes} Cities - {self.tests} Graphs')
        plt.xticks([i for i in range(self.tests)])
        plt.legend()
        fig = plt.gcf()
        plt.show()
        plt.draw()
        fig.savefig(f'{self.nodes} Cities - {self.tests} Tests.png')

        if self.extended:
            for key in self.dataExt.keys():
                plt.plot(self.dataExt[key], label = key)
            plt.title(f'{self.nodes} Cities - {self.tests} Graphs Extended')
            plt.xticks([i for i in range(self.tests)])
            plt.legend()
            fig = plt.gcf()
            plt.show()
            plt.draw()
            fig.savefig(f'{self.nodes} Cities - {self.tests} Tests Ext.png')
    
    def ratio_G_Ch(self):
        """
        Returns the mean Greedy2Ext to Christofides ratio
        """
        greedy_christofieds_ratio = []
        for i in range(self.tests):
            ratio = self.data["Greedy2Ext"][i]/self.data["Christofides"][i]
            greedy_christofieds_ratio.append(ratio)
        return np.mean(greedy_christofieds_ratio)




def main():

    # The tests are just example tests 
    # like those I used to generate data for the report
    # If run they will generate new graphs and
    # the resultant plots will be different from the ones in the report
    # The tests can be run by uncommenting the last line in the if statement

    test1 = Test1(nodes = 50)
    test1.findTour()
    test1.plot()

    test2 = Test2(nodes = 10, extended=True)
    test2.setUp()
    test2.findTour()
    print(f"Greedy2Ext/Christofides ratio: {test2.ratio_G_Ch()}")
    print(test2.ratio())
    test2.plot()

if __name__ == "__main__":
    main()