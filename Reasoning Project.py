import heapq
import math


class Node:
    id = None
    up = None
    down = None
    left = None
    right = None
    previousNode = None
    edgeCost = None
    gOfN = None  # total edge cost
    hOfN = None  # heuristic value
    heuristicFn = None
    x = None
    y = None

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class SearchAlgorithms:
    Path = []
    fullPath = []
    totalCost = -1

    def __init__(self, mazeStr, cost=None):
        self.maze = self.mazeStr_to_matrix(mazeStr)
        self.cost = cost

        if self.cost:
            for i in range(len(self.maze)):
                for j in range(len(self.maze[i])):
                    self.maze[i][j].cost = self.cost[i * len(self.maze[i]) + j]

    def mazeStr_to_matrix(self, mazeStr):
        # Split the maze string by spaces to separate rows
        maze = mazeStr.split(" ")
        # Initialize an empty matrix to represent the maze
        matrix = []
        # Iterate over each row in the maze
        for i in range(len(maze)):
            # Split the row by commas to separate individual nodes
            row = maze[i].split(",")
            # Initialize an empty list to store nodes in the current row
            node_row = []
            # Iterate over each node in the row
            for index, j in enumerate(row):
                # Create a Node object with the value of the current node
                node = Node(j)
                # Set a unique ID for each node based on its position in the matrix
                node.id = i * len(row) + index
                # Append the node to the current row
                node_row.append(node)
                # Check if the node is the start node
                if node.value == "S":
                    # If so, set it as the start node for the maze
                    self.start = node
                # Check if the node is the end node
                if node.value == "E":
                    # If so, set it as the end node for the maze
                    self.end = node
            # Append the row of nodes to the matrix
            matrix.append(node_row)

        # Set up, down, left, and right connections for each node
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                # Set the x and y coordinates for the current node
                matrix[i][j].x = i
                matrix[i][j].y = j
                # Check if there's a node above the current node and it's not a wall
                if i > 0 and matrix[i - 1][j].value != "#":
                    # If so, set it as the up connection for the current node
                    matrix[i][j].up = matrix[i - 1][j]
                # Check if there's a node below the current node and it's not a wall
                if i < len(matrix) - 1 and matrix[i + 1][j].value != "#":
                    # If so, set it as the down connection for the current node
                    matrix[i][j].down = matrix[i + 1][j]
                # Check if there's a node to the left of the current node and it's not a wall
                if j > 0 and matrix[i][j - 1].value != "#":
                    # If so, set it as the left connection for the current node
                    matrix[i][j].left = matrix[i][j - 1]
                # Check if there's a node to the right of the current node and it's not a wall
                if j < len(matrix[i]) - 1 and matrix[i][j + 1].value != "#":
                    # If so, set it as the right connection for the current node
                    matrix[i][j].right = matrix[i][j + 1]

        # Return the constructed matrix representing the maze
        return matrix

    def DFS(self):
        # Clear the path, full path, and total cost variables
        self.Path.clear()
        self.fullPath.clear()
        self.totalCost = 0

        # Initialize a stack with the start node and its path
        stack = [(self.start, [self.start.id])]
        # Initialize a stack to keep track of visited node IDs
        stack_id = [self.start.id]

        # Initialize a set to store visited node IDs
        visited = set()

        # Find the end node in the maze
        for row in self.maze:
            for node in row:
                if node.value == "E":
                    self.end_node = node
                    break

        # Perform depth-first search until the stack is empty
        while stack:
            # Pop the node and its path from the stack
            node, path = stack.pop()
            stack_id.pop()
            # Append the current node's ID to the full path
            self.fullPath.append(node.id)

            # Check if the current node is the end node
            if node.value == "E":
                # If so, set the path to the end node and break out of the loop
                self.Path = path
                break

            # Check if the current node has not been visited and is not a wall
            if node.id not in visited and node.value != "#" and node.id not in stack_id:
                # Add the current node to the set of visited nodes
                visited.add(node.id)

                # Explore neighboring nodes
                neighbors = []
                for direction in ["right", "left", "down", "up"]:
                    # Get the neighbor of the current node in the specified direction
                    neighbor = getattr(node, direction)

                    # Check if the neighbor exists, has not been visited, is not a wall, and is not already in the stack
                    if (
                        neighbor
                        and neighbor.id not in visited
                        and neighbor.value != "#"
                        and neighbor.id not in stack_id
                    ):
                        # Add the neighbor and its path to the list of neighbors
                        neighbors.append((neighbor, path + [neighbor.id]))

                # Add neighbors to the stack
                for neighbor in neighbors:
                    stack.append(neighbor)
                    stack_id.append(neighbor[0].id)

        # Return the path and full path
        return self.Path, self.fullPath


    def BFS(self):
        # Clear the path, full path, and total cost variables
        self.Path.clear()
        self.fullPath.clear()
        self.totalCost = 0

        # Initialize a queue with the start node and its path
        queue = [(self.start, [self.start.id])]
        # Initialize a set to store visited node IDs
        visited = set()

        # Perform breadth-first search until the queue is empty
        while queue:
            # Dequeue the node and its path from the queue
            node, path = queue.pop(0)

            # Check if the current node is the end node
            if node.value == "E":
                # If so, set the path to the end node and break out of the loop
                self.Path = path
                # Optionally calculate total cost (length of path minus 1)
                # self.totalCost = len(path) - 1  # Subtract 1 because the start node doesn't count towards the cost
                break

            # Check if the current node has not been visited
            if node.id not in visited:
                # Mark the current node as visited
                visited.add(node.id)
                # Append the current node's ID to the full path
                self.fullPath.append(node.id)

                # Explore neighboring nodes
                for direction in ["up", "down", "left", "right"]:
                    # Get the neighbor of the current node in the specified direction
                    neighbor = getattr(node, direction)
                    # Check if the neighbor exists, has not been visited, and is not a wall
                    if neighbor and neighbor.id not in visited and neighbor.value != "#":
                        # Enqueue the neighbor and its path to the queue
                        queue.append((neighbor, path + [neighbor.id]))

        # Append the last node of the path to the full path
        self.fullPath.append(self.Path[-1])

        # Return the path and full path
        return self.Path, self.fullPath

    def UCS(self):
            # Clear the fullPath, Path lists and totalCost
            self.fullPath.clear()
            self.Path.clear()
            self.totalCost = 0

            # Initialize the priority queue with the start node
            queue = [(0, self.start, [self.start.id])]  # Store the cost, node and the path
            visited = set()

            while queue:
                cost, node, path = heapq.heappop(queue) 
                self.fullPath.append(node.id)  # Append the node's id

                # If the node is the goal, append the path to the Path list and update totalCost
                if node.value == "E":
                    self.Path = path
                    self.totalCost = cost
                    break

                if node not in visited:
                    visited.add(node)

                    # Add the node's neighbors to the queue
                    for direction in ["right", "down", "up", "left"]:
                        neighbor = getattr(node, direction)
                        if neighbor and neighbor not in visited and neighbor.value != "#":
                            heapq.heappush( 
                                queue,
                                (cost + neighbor.cost, neighbor, path + [neighbor.id]),
                            )

            return self.Path, self.fullPath

    def AstarEuclideanHeuristic(self):
        # Clear the fullPath, Path, and totalCost variables
        self.fullPath.clear()
        self.Path.clear()
        self.totalCost = 0

        # Initialize a priority queue with the start node and its path, with priority based on heuristic cost
        queue = [(0, 0, self.start.id, self.start, [self.start.id])]  # (heuristic cost, cost, node_id, node, path)
        visited = set()  # Initialize a set to store visited nodes

        # Find the goal node
        goal_node = None
        for row in self.maze:
            for node in row:
                if node.value == "E":
                    goal_node = node
                    break

        # Perform A* search until the queue is empty
        while queue:
            # Pop the node with the lowest combined cost and heuristic from the priority queue
            heuristic, cost, node_id, node, path = heapq.heappop(queue)
            # Append the current node's ID to the fullPath
            self.fullPath.append(node.id)

            # Check if the current node is the goal node
            if node == goal_node:
                # If so, set the path, total cost, and break out of the loop
                self.Path = path
                self.totalCost = cost
                break

            # Check if the current node has not been visited
            if node not in visited:
                # Mark the current node as visited
                visited.add(node)

                # Explore neighboring nodes
                for direction in ["right", "down", "up", "left"]:
                    # Get the neighbor of the current node in the specified direction
                    neighbor = getattr(node, direction)
                    # Check if the neighbor exists, has not been visited, and is not a wall
                    if neighbor and neighbor not in visited and neighbor.value != "#":
                        # Calculate the cost to reach the neighbor from the start node
                        neighbor_cost = cost + neighbor.cost
                        # Calculate the Euclidean heuristic from the neighbor to the goal node
                        neighbor_heuristic = math.sqrt(
                            (neighbor.x - goal_node.x) ** 2
                            + (neighbor.y - goal_node.y) ** 2
                        )
                        # Push the neighbor to the priority queue with priority based on combined cost and heuristic
                        heapq.heappush(
                            queue,
                            (
                                neighbor_heuristic + neighbor_cost,
                                neighbor_cost,
                                neighbor.id,
                                neighbor,
                                path + [neighbor.id],
                            ),
                        )

        # Return the path and fullPath
        return self.Path, self.fullPath


    def AstarManhattanHeuristic(self):
        # Clear the fullPath, Path, and totalCost variables
        self.fullPath.clear()
        self.Path.clear()
        self.totalCost = 0

        # Initialize a queue with the start node and its path, with priority based on Manhattan heuristic
        queue = [(0, 0, self.start.id, self.start, [self.start.id])]
        queue_set = {self.start}  # Initialize a set to keep track of nodes in the queue
        visited = set()  # Initialize a set to store visited nodes

        # Find the goal node
        goal_node = None
        for row in self.maze:
            for node in row:
                if node.value == "E":
                    goal_node = node
                    break

        # Perform A* search until the queue is empty
        while queue:
            # Dequeue the node with the lowest combined cost and heuristic from the queue
            heuristic, cost, node_id, node, path = queue.pop(0)
            queue_set.remove(node)  # Remove the node from the queue set
            # Append the current node's ID to the fullPath
            self.fullPath.append(node.id)

            # Check if the current node is the goal node
            if node == goal_node:
                # If so, set the path, total cost, and break out of the loop
                self.Path = path
                self.totalCost = cost
                break

            # Check if the current node has not been visited
            if node not in visited:
                # Mark the current node as visited
                visited.add(node)

                # Explore neighboring nodes
                for direction in ["down", "right", "up", "left"]:
                    # Get the neighbor of the current node in the specified direction
                    neighbor = getattr(node, direction)
                    # Check if the neighbor exists, has not been visited, is not a wall, and is not already in the queue
                    if (
                        neighbor
                        and neighbor not in visited
                        and neighbor.value != "#"
                        and neighbor not in queue_set
                    ):
                        # Calculate the cost to reach the neighbor from the start node
                        neighbor_cost = cost + 1
                        # Calculate the Manhattan heuristic from the neighbor to the goal node
                        neighbor_heuristic = abs(neighbor.x - goal_node.x) + abs(
                            neighbor.y - goal_node.y
                        )
                        # Enqueue the neighbor with priority based on combined cost and heuristic
                        queue.append(
                            (
                                neighbor_heuristic + neighbor_cost,
                                neighbor_cost,
                                neighbor.id,
                                neighbor,
                                path + [neighbor.id],
                            )
                        )
                        # Add the neighbor to the queue set
                        queue_set.add(neighbor)

        # Return the path and fullPath
        return self.Path, self.fullPath


def main():
    s1 = SearchAlgorithms(
        "S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,."
    )
    path, fullPath = s1.BFS()
    print("BFS Path: " + str(path), end="\nFull Path is: ")
    print(fullPath)

    s2 = SearchAlgorithms(
        "S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,."
    )

    path, fullPath = s2.DFS()
    print("DFS Path: " + str(path), end="\nFull Path is: ")
    print(fullPath)

    s3 = SearchAlgorithms(
        "S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.",
        [0, 15, 2, 100, 60, 35, 30, 3
         , 100, 2, 15, 60, 100, 30, 2
         , 100, 2, 2, 2, 40, 30, 2, 2
         , 100, 100, 3, 15, 30, 100, 2
         , 100, 0, 2, 100, 30])
    path, fullPath = s3.UCS()
    print("UCS Path: " + str(path), end="\nFull Path is: ")
    print(fullPath)
    print("Total Cost: " + str(s3.totalCost))

    s4 = SearchAlgorithms(
        "S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.",
        [0, 15, 2, 100, 60, 35, 30, 3
         , 100, 2, 15, 60, 100, 30, 2
         , 100, 2, 2, 2, 40, 30, 2, 2
         , 100, 100, 3, 15, 30, 100, 2
         , 100, 0, 2, 100, 30])
    path, fullPath = s4.AstarEuclideanHeuristic()
    print("AstarEcludianHeuristic Path: " + str(path), end="\nFull Path is: ")
    print(fullPath)
    print("Total Cost: " + str(s4.totalCost))

    s5 = SearchAlgorithms(
        "S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,."
    )
    path, fullPath = s5.AstarManhattanHeuristic()
    print("AstarManhattanHeuristic Path: " + str(path), end="\nFull Path is: ")
    print(fullPath)
    print("Total Cost: " + str(s5.totalCost))


main()
