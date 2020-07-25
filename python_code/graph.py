import math
import copy
import sys
import time

import Christofides

def euclid(p,q):
    x = p[0]-q[0]
    y = p[1]-q[1]
    return math.sqrt(x*x+y*y)
                
class Graph:

    def __init__(self,n,filename):
        file = open(filename, "r")
        self.n = n

        # Reading the file and parse input as float numbers
        lines = []
        for l in file:
            lines.append(l.split())
            for j in range(len(lines[-1])):
                if n == -1:
                    lines[-1][j] = float(lines[-1][j])
                else:
                    lines[-1][j] = int(lines[-1][j])


        if n == -1:
            # Euclidean TSP
            self.dist = [[0 for j in range(len(lines))] for i in range(len(lines))]
            self.n = len(lines)
            for i in range(len(lines)):
                for j in range(len(lines)):
                    self.dist[i][j] = euclid(lines[i], lines[j])

        elif n>0:
            # Metric TSP
            self.dist = [[0 for j in range(self.n)] for i in range(self.n)]
            for i in range(len(lines)):
                self.dist[lines[i][0]][lines[i][1]] = lines[i][2]
                self.dist[lines[i][1]][lines[i][0]] = lines[i][2]
        
        self.perm = [i for i in range(self.n)]
        # Initializing additional 2D array for Dynamic Programming Algorithm
        # Setting a node limit due to exponential time 
        if self.n <= 15:
            self.pathLen = [[(-1) for j in range(1<<self.n)] for i in range(self.n)]


    def tourValue(self):
        """
        Calculates the cost of a whole tour given the node premutation
        """
        value = 0
        temp_perm = self.perm.copy()
        temp_perm.append(temp_perm[0])
        for i in range(len(self.perm)):
            value +=self.dist[temp_perm[i]][temp_perm[i+1]]
        return value


    def trySwap(self,i):
        """
        Swaps the position of two adjacent cities in the permutation
        if they lead to lower tour cost
        """
        current_dist = self.dist[(self.perm[(i-1) % self.n])][self.perm[i]] + self.dist[self.perm[(i+1) % self.n]][self.perm[(i+2) % self.n]]
        trial_dist = self.dist[self.perm[(i-1) % self.n]][self.perm[(i+1) % self.n]] + self.dist[self.perm[i]][self.perm[(i+2) % self.n]]
        if current_dist > trial_dist:
            # Swap is successful (reduces the cost of current tour)
            # Swapping the positions
            temp = self.perm[i]
            self.perm[i] = self.perm[(i+1) % self.n]
            self.perm[(i+1) % self.n] = temp
            return True
        else:
            return False


    def tryReverse(self,i,j):
        """
        Reverses a whole sequence i-j
        if it leads to lower tour cost
        """
        start = self.perm[(i-1) % self.n]
        end = self.perm[(j+1) % self.n]

        current_dist = self.dist[start][self.perm[i]] + self.dist[self.perm[j]][end]
        trial_dist = self.dist[start][self.perm[j]] + self.dist[self.perm[i]][end]

        if current_dist > trial_dist:
            # Reverse is successful (reduces the cost of current tour)
            # Reversing the splice
            self.perm[i:j+1] = self.perm[i:j+1][::-1]
            return True
        else:
            return False

    # Predefined function
    def swapHeuristic(self):
        better = True
        while better:
            better = False
            for i in range(self.n):
                if self.trySwap(i):
                    better = True

    # Predefined function
    def TwoOptHeuristic(self):
        better = True
        while better:
            better = False
            for j in range(self.n-1):
                for i in range(j):
                    if self.tryReverse(i,j):
                        better = True


    def Greedy(self):
        """
        Greedy algorithm building a tour:
        starts from node 0;
        takes closest not visited node as next
        """
        self.perm[0] = 0
        #list of unused nodes
        idx = [i for i in range(1,self.n)]
        # constructing the tour by appending the next closest node on each iteration
        for i in range(self.n-1):
            # list of tuples holding the node and the distance to that node
            distances = map(lambda x: (x, self.dist[self.perm[i]][x]), idx)
            min_dist = min(distances, key = lambda x: x[1])
            self.perm[i+1]= min_dist[0]
            idx.remove(min_dist[0])

    def _printTree(self, T, V):
        """
        Helper function used for debugging:
        Parameters:
        - T - an array of edges represented as tuples
        - V - an array of vertices in the tree

        Prints:
        - edges of a tree
        - vertices
        - total weight of the tree
        """

        print(f"Vertices: {V}")
        print(f"Edges: {T}")
        value = 0
        for edge in T:
            value+=self.dist[edge[0]][edge[1]]
        print(f"Cost: {value}")

    # Helper functions for Christofides Algorithm

    def _minKey(self, key, V):
        """
        Helper function for primMST:
        finds a vertex not included in the tree but closest to the tree
        """
        min = sys.maxsize

        for v in range(self.n):
            if key[v] < min and not(v in V):
                min = key[v]
                min_index = v
        return min_index
    
    def _primMST(self):
        """
        Prim's algorithm for generating:
        Minimum Spanning Tree

        Returns:
        - T - tree of tuples as edges
        """

        # key holds the min distance from the vertices already in the tree
        # and those not in the tree (the cut)
        key = [sys.maxsize for i in range(self.n)]
        parent = [None for i in range(self.n)]
        #First vertex is picked as the first in the MST
        key[0] = 0
        
        T = [] # Array of tuples, holding the edges in the tree
        V = [] # Array of vertices in the tree

        for i in range(self.n):

            next_vertex = self._minKey(key, V) # cloest vertex to the tree
            V.append(next_vertex)

            # Updating the values of key and parent given the new vertex in the tree
            for v in range(self.n):
                if v != next_vertex and not(v in V) and key[v] > self.dist[next_vertex][v]:
                    key[v] = self.dist[next_vertex][v]
                    parent[v] = next_vertex
        
        for i,v in enumerate(parent):
            # v is None for the first vertex
            if v != None:
                T.append((i,v))

        return T

    def _oddDegree(self, T):
        """
        given a tree graph (array of (self.n -1) edges as tuples)
        returns an array of the nodes of odd degree
        """
        # 'unzipping' the edges of T
        nodes = [i for i,j in T] + [j for i,j in T]
        
        #stores tuples of a node and its degree
        sorted = []
        while len(nodes) > 0:
            nodeDeg = (nodes[0], nodes.count(nodes[0]))
            sorted.append(nodeDeg)
            nodes = [ node for node in nodes if node != nodeDeg[0]]
        odd = [node for (node, deg) in sorted if deg % 2 == 1]
        # taking the nodes of odd degree
        # filtering to get only odd degree nodes and extracting the node

        return odd
    
    def _matching(self, V):
        """
        given a list of vertices (even number of vertices)
        finds and returns a minimum weight matching as an array of edges (tuples)
        Greedy heuristic algorithm - does not guarantee ultimate minimum wight
        """
        matching_edges = []

        while len(V) != 0:
            u = V.pop()

            distances = [(node, self.dist[u][node]) for node in V]
            min_dist = min(distances, key= lambda x: x[1])
            matching_edges.append((u, min_dist[0]))
            V.remove(min_dist[0])
        return matching_edges


    def _addEdge(self, edge):
        self.neighbours[edge[0]].append(edge[1])
        self.neighbours[edge[1]].append(edge[0])

    def _removeEdge(self, edge):
        self.neighbours[edge[0]].remove(edge[1])
        self.neighbours[edge[1]].remove(edge[0])


    def _DFSCount(self, v, visited):
        """
        Counts number of reachable nodes starting from node v
        Takes into account already visited nodes
        """
        count = 1
        visited[v] = True

        for node in self.neighbours[v]:
            if visited[node] == False:
                count = count + self._DFSCount(node, visited)
        return count
    
    def _isValidEdge(self, edge):
        """
        An edge is considered valid
        if removing it does not reduce the number of nodes reachable from its start edge

        Parameter:
        - edge - a tuple of nodes
        """
        if len(self.neighbours[edge[0]]) == 1:
            return True
        
        else:
            visited = [False for i in range(self.n)]
            # holds the number of reachable nodes
            count1 = self._DFSCount(edge[0], visited)

            self._removeEdge(edge)
            visited = [False for i in range(self.n)]
            # holds number of reachable nodes after edge removal
            count2 = self._DFSCount(edge[0], visited)
            self._addEdge(edge)

            if count1 > count2:
                return False
            else:
                return True

    def _circuit(self):
        """
        Recursive function for constructing Euler circuit
        """
        for neighbour in self.neighbours[self.path[-1]]:
            if self._isValidEdge((self.path[-1], neighbour)):
                self._removeEdge((self.path[-1], neighbour))
                self.path.append(neighbour)
                self._circuit()

    def _eulerCircuit(self, T):
        """
        Given a valid Eulerian graph returns an Eulerian Circuit
        
        Parameter:
        - T - array of tuples (edges of graph)
        """
        # dictionary to hold the neighbours of each node in the graph
        self.neighbours = {}
        
        for edge in T:
            if edge[0] not in self.neighbours:
                self.neighbours[edge[0]] = []

            if edge[1] not in self.neighbours:
                self.neighbours[edge[1]] = []
            
            self._addEdge(edge)

        # path holds the euler path with start node
        self.path = [T[0][0]]
        self._circuit()
        return self.path

    def _hamiltonianCircuit(self):
        """
        Given the self.path containing the Euler Circuit,
        generates a hamiltoian circuit by shortcutting:
        Skips already visited nodes
        Assigns the path to self.perm
        """
        hamiltonian = [(idx, node) for idx, node in enumerate(self.path)]
        for idx, node in enumerate(self.path):
            if node in self.path[0:idx]:
                hamiltonian.remove((idx, node))
        hamiltonian = [node for (idx, node) in hamiltonian]

        self.perm = hamiltonian

    def Christfides(self):
        """
        Complete algorithm using predefinded functions
        """
        T = self._primMST()
        odd = self._oddDegree(T)
        matching = self._matching(odd)
        self._eulerCircuit(T + matching)
        self._hamiltonianCircuit()

    def _partialtourValue(self, path):
        """
        Calculates tourValue of a given path
        """
        cost = 0
        for i in range(len(path)):
            cost += self.dist[path[i]][path[(i+1) % len(path)]]
        return cost

    def Greedy2(self, start_node = 0):
        """
        Constructs a permutation
        by adding the nodes to the path in an optimal position
        Assigns path to self.perm
        """
        visited = [False] * self.n
        path = [start_node]
        visited[start_node] = True

        for node in range(0, self.n):
            if visited[node] == False:
                min_cost = sys.maxsize
                visited[node] = True

                for idx in range(node + 1):
                    path.insert(idx, node)
                    # checks if the current path is optimal
                    if self._partialtourValue(path) < min_cost:
                        best_path = path[:]
                        min_cost = self._partialtourValue(path)
                    path.pop(idx)
                path = best_path[:]

        self.perm = best_path[:]
        return  best_path

    def Greedy2Ext(self):
        """
        Runs Greedy2
        with every node as starting node
        Assigns optimal path to self.perm
        """
        min_cost = sys.maxsize
        for node in range(self.n):
            curr_path = self.Greedy2(node)
            curr_cost = self.tourValue()
            if curr_cost < min_cost:
                min_cost = curr_cost
                min_path = curr_path[:]
        self.perm = min_path[:]

    def DPAlgorithm(self, visited = 1, pos = 0, start = 0):
        """
        Held-Karp
        Dynamic Programming recursive algorithm (exponential time)
        Used on smaller sets of nodes for experiments

        Parameters:
        - visited - range 0-(2^n-1), binary number denoting
                    if a node is visited: i'th node is visited if i'th bit is 1, 0 - otherwise
        - pos - the node/city which is visited last
        - start - the start node (does not change throughout the iterations)
        """
        if self.n > 15:
            return False
        
        all_visited = (1<<self.n) - 1

        # base case
        if (visited == all_visited):
            return self.dist[pos][start]

        if(self.pathLen[pos][visited] != -1):
            # already calculated cost
            return self.pathLen[pos][visited]

        cost = sys.maxsize

        for node in range(self.n):
            # using bitwise operations to check if a certain node is visited
            if (visited & (1 << node) == 0):
                new_cost = self.dist[node][pos] + self.DPAlgorithm((visited|(1<<node)), node, start)
                cost = min(cost, new_cost)
        self.pathLen[pos][visited] = cost
        return cost

    def DPAlgorithmExt(self):
        """
        Finds the solution of TSP
        by setting each node to be a starting node 
        returns the minimal cost
        """
        if self.n > 15:
            return False
        min_cost = sys.maxsize
        for node in range(self.n):
            curr_cost = self.DPAlgorithm((1<<node), node, node)
            if min_cost > curr_cost:
                min_cost = curr_cost
        return min_cost