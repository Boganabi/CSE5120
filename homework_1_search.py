from queue import PriorityQueue, Queue
from pprint import pprint

# Graph of cities with connections to each city. Similar to our class exercises, you can draw it on a piece of paper 
# with step-by-step node inspection for better understanding 
graph = {
    'San Bernardino': ['Riverside', 'Rancho Cucamonga'],
    'Riverside': ['San Bernardino', 'Ontario', 'Pomona'],
    'Rancho Cucamonga': ['San Bernardino', 'Azusa', 'Los Angeles'],
    'Ontario': ['Riverside', 'Whittier', 'Los Angeles'],
    'Pomona': ['Riverside', 'Whittier', 'Azusa', 'Los Angeles'],
    'Whittier': ['Ontario','Pomona', 'Los Angeles'],
    'Azusa': ['Rancho Cucamonga', 'Pomona', 'Arcadia'],
    'Arcadia': ['Azusa', 'Los Angeles'],
    'Los Angeles': ['Rancho Cucamonga', 'Ontario', 'Pomona', 'Whittier', 'Arcadia']
}

# Weights are treated as g(n) function as we studied in our class lecture which represents the backward cost. 
# In the data structure below, the key represents the cost from a source to target node. For example, the first
# entry shows that there is a cost of 2 for going from San Bernardino to Riverside.
weights = {
    ('San Bernardino', 'Riverside'): 2,
    ('San Bernardino', 'Rancho Cucamonga'): 1,
    ('Riverside', 'Ontario'): 1,
    ('Riverside', 'Pomona'): 3,
    ('Rancho Cucamonga', 'Los Angeles'): 5,
    ('Pomona', 'Los Angeles'): 2,
    ('Ontario', 'Whittier'): 2,
    ('Ontario', 'Los Angeles'): 3,
    ('Rancho Cucamonga', 'Azusa'): 3,
    ('Pomona', 'Azusa'): 2,
    ('Pomona', 'Whittier'): 2,
    ('Azusa', 'Arcadia'): 1,
    ('Whittier', 'Los Angeles'): 2,
    ('Arcadia', 'Los Angeles'): 2
}

# heurist is the h(n) function as we studied in our class lecture which represents the forward cost. 
# In the data structure below, each entry represents the h(n) value. For example, the second entry
# shows that h(Riverside) is 2 (i.e., h value as forward cost for eaching at Riverside assuming that
# your current/start city is San Bernardino)

heuristic = {
    'San Bernardino': 4,
    'Riverside': 2,
    'Rancho Cucamonga': 1,
    'Ontario': 1,
    'Pomona': 3,
    'Whittier': 4,
    'Azusa': 3,
    'Arcadia': 2,
    'Los Angeles': 0
}

# Data structure to implement search algorithms. Each function below currently has one line of code
# returning empty solution with empty expanded cities. You can remove the current return statement and 
# implement your code to complete the functions.

class SearchAlgorithms:
    def breadthFirstSearch(self, start, goal, graph):
        """
        Search the shallowest nodes in the search tree first.

        Your search algorithm needs to return (i) a list of cities the algorithm will propose to go to to reach the
        goal, and (ii) set of expanded cities (visited nodes). Make sure to implement a graph search algorithm.

        """
        "*** YOUR CODE HERE ***"

        if start == goal: return "Returned solution:", start, "Expanded cities:", start

        q = Queue()
        expanded_cities = set() # use set since constant lookup time

        # add inital node to queue since we have to use it to start and mark as visited
        # this may look like a weird insertion, but we are inserting a tuple. first is the actual node visted, second is the path represented as a list
        # the second part of the tuple is important, as it shows the fastest path to the current node we are exploring
        q.put((start, [start]))
        expanded_cities.add(start)

        while q.empty() == False: 
            curr_node, possible_solution = q.get() # possible_solution now holds the current path to get here
            for city in graph[curr_node]: # iterate through all neighbors
                if city not in expanded_cities: # check if not explored to avoid loops
                    expanded_cities.add(city) # add current city to set of already shown cities
                    if city == goal:
                        solution = possible_solution + [city]
                        return "Returned solution:", solution, "Expanded cities:", expanded_cities
                    # insert city and path to get here
                    q.put((city, possible_solution + [city]))
                    
        # You can delete the line below once you have implemented your solution above
        return {"Returned solution: [], Expanded cities: []"}

    def depthFirstSearch(self, start, goal, graph):
        """
        Search the deepest nodes in the search tree first.

        Your search algorithm needs to return (i) a list of cities the algorithm will propose to go to to reach the
        goal, and (ii) set of expanded cities (visited nodes). Make sure to implement a graph search algorithm.

        Please be very careful when you expand the neighbor nodes in your code when using stack. In case of using 
        normal list or a data structure other than the Stack, you might need to reverse the order of the neighbor nodes
        before you push them in the stack to get correct results 

        """
        "*** YOUR CODE HERE ***"

        if start == goal: return "Returned solution:", start, "Expanded cities:", start

        stack = []
        expanded_cities = set() # use set since constant lookup time

        stack.append((start, [start])) # similar way of storing path as in BFS

        while len(stack) != 0:
            curr_city, path = stack.pop() # very similar process to BFS
            if curr_city not in expanded_cities:
                expanded_cities.add(curr_city)
                if curr_city == goal:
                    return "Returned solution:,", path, ", Expanded cities:", expanded_cities
                for city in reversed(graph[curr_city]): # since stack reads last in, in order to read the correct node we need to put the first node last
                    stack.append((city, path + [city]))

        # You can delete the line below once you have implemented your solution above
        return {"Returned solution: [], Expanded cities: []"}
            
    def uniformCostSearch(self, start, goal, graph, weights):
        """Search the node of least total cost first.
        Important things to remember
        1 - Use PriorityQueue with .put() and .get() functions
        2 - In addition to putting the start or current node in the queue, also put the cost (g(n)) using weights data structure
        3 - When you're expanding the neighbor of the current you're standing at, get its g(neighbor) by weights[(node, neighbor)] 
        4 - Calling weights[(node, neighbor)] may throw KeyError exception which is due to the fact that the weights data structure
            only has one directional weights. In the class, we mentioned that there is a path from Arad to Sibiu and back. If the 
            exception occurs, you will need to get the weight of the nodes in reverse direction (weights[(neighbor, node)])
        """

        "*** YOUR CODE HERE ***"

        if start == goal: return "Returned solution:", start, "Expanded cities:", start

        q = PriorityQueue()
        expanded_cities = set() # use set since constant lookup time

        q.put((0, start, [start])) # again similar to the tuple storage used in BFS and DFS, but now we also have a priority with it

        while(q.not_full): # similar logic to BFS and DFS
            node = q.get()
            # unpack current item in queue
            cost = node[0]
            curr_city = node[1]
            path = node[2]
            if curr_city == goal:
                solution = path
                return "Returned solution:", solution, "Expanded cities:", expanded_cities
            expanded_cities.add(curr_city)
            for city in graph[curr_city]:
                if city not in expanded_cities:
                    new_cost = 0
                    # allows us to get either direction for the index in the weights dictionary
                    try:
                        new_cost = weights[(curr_city, city)]
                    except KeyError:
                        new_cost = weights[(city, curr_city)]
                    q.put((cost + new_cost, city, path + [city]))

        # You can delete the line below once you have implemented your solution above
        return {"Returned solution: [], Expanded cities: []"}

    def AStar(self, start, goal, graph, weights, heuristic):
        """Search the node that has the lowest combined cost and heuristic first.
        Important things to remember
        1 - Use PriorityQueue with .put() and .get() functions
        2 - In addition to putting the start or current node in the queue, and the g(n), also put the combined cost (i.e., g(n) + h(n)) 
            using weights and heuristic data structure
        3 - When you're expanding the neighbor of the current you're standing at, get its g(neighbor) by weights[(node, neighbor)] 
        4 - Calling weights[(node, neighbor)] may throw KeyError exception which is due to the fact that the weights data structure
            only has one directional weights. In the class, we mentioned that there is a path from Arad to Sibiu and back. If the 
            exception occurs, you will need to get the weight of the nodes in reverse direction (weights[(neighbor, node)])
        """
        "*** YOUR CODE HERE ***"

        # A* is very similar to UCS code-wise, only need to add heuristic to queue priority i think

        if start == goal: return "Returned solution:", start, "Expanded cities:", start

        q = PriorityQueue()
        expanded_cities = set() # use set since constant lookup time

        q.put((0 + heuristic[start], start, [start])) # again similar to the tuple storage used in BFS and DFS, but now we also have a priority with it

        while(q.not_full): # similar logic to BFS and DFS
            node = q.get()
            # unpack current item in queue
            cost = node[0]
            curr_city = node[1]
            path = node[2]
            if curr_city == goal:
                solution = path
                return "Returned solution:", solution, "Expanded cities:", expanded_cities
            expanded_cities.add(curr_city)
            for city in graph[curr_city]:
                if city not in expanded_cities:
                    new_cost = 0
                    # allows us to get either direction for the index in the weights dictionary
                    try:
                        new_cost = weights[(curr_city, city)]
                    except KeyError:
                        new_cost = weights[(city, curr_city)]
                    q.put((cost + new_cost + heuristic[city], city, path + [city]))

        # You can delete the line below once you have implemented your solution above
        return {"Returned solution: [], Expanded cities: []"}

# Call to create the object of the above class
search = SearchAlgorithms()

# Call to each algorithm to print the results
print("Breadth First Search Result") # ['San Bernardino', 'Rancho Cucamonga', 'Los Angeles']
pprint(search.breadthFirstSearch('San Bernardino', 'Los Angeles', graph))

print("Depth First Search Result") # ['San Bernardino', 'Riverside', 'Ontario', 'Whittier', 'Pomona', 'Azusa', 'Rancho Cucamonga', 'Los Angeles']
pprint(search.depthFirstSearch('San Bernardino', 'Los Angeles', graph))

print("Uniform Cost Search Result") # ['San Bernardino', 'Rancho Cucamonga', 'Los Angeles']
pprint(search.uniformCostSearch('San Bernardino', 'Los Angeles', graph, weights))

print("A* Search Result") # ['San Bernardino', 'Rancho Cucamonga', 'Los Angeles']
pprint(search.AStar('San Bernardino', 'Los Angeles', graph, weights, heuristic))
