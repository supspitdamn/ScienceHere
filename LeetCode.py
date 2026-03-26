# class LinkedList:

#     class Node:

#         def __init__(self, value, next):

#             self.value = value
#             self.next = next
        
#         def __lt__(self, other):

#             return self.value < other.value
        
#         def __eq__(self, other):

#             return self.value == other.value
    
#     def __init__(self, *args):

#         self.__Head = None
#         self.__Tail = None
#         self.__length = 0

#         for x in args:

#             self.push(x)
    
#     def push(self, value):

#         if self.__Head is None:

#             self.__Head = self.Node(value, None)
#             self.__Tail = self.__Head
        
#         else:

#             self.__Tail.next = self.Node(value, None)
#             self.__Tail = self.__Tail.next
        
#         self.__length += 1
    
#     def __str__(self):
#         runner = self.__Head
#         lst = []
#         while runner is not None:
#             lst.append(str(runner.value))
#             runner = runner.next
#         return "->".join(lst)
    
#     def __getitem__(self, index):


#         runner = self.__Head
#         if runner is None:
#             return None

#         try:
#             for _ in range(index):
#                 if runner is None:
#                     raise IndexError
#                 runner = runner.next

#         except (IndexError, AttributeError):
#             return None
        
#         return runner.value
    
#     def drop_duplicates(self):

#         result = LinkedList()
#         runner = self.__Head

#         while runner is not None:

#             if result.__length == 0 or result.__Tail != runner:

#                 result.push(runner.value)
            
#             runner = runner.next
        
#         return result

# lst = LinkedList(1,2,2,2,3,3,4,4,4,4,5,5)
# lst1 = lst.drop_duplicates()
# print(lst1)

# # num1 = sorted(list(map(int, input().split())))
# # num2 = sorted(list(map(int, input().split())))

# # def merge(n1, n2):
# #     temp = n1.copy()
# #     res = []
# #     it1, it2 = 0, 0

# #     # 1. Сравниваем элементы
# #     while it1 < len(temp) and it2 < len(n2):
# #         if temp[it1] < n2[it2]:
# #             res.append(temp[it1])
# #             it1 += 1
# #         else:
# #             res.append(n2[it2])
# #             it2 += 1

# #     # 2. Добавляем остатки (обязательно!)
# #     res.extend(temp[it1:])
# #     res.extend(n2[it2:])

# #     # 3. Главная магия: меняем num1 "на месте"
# #     n1[:] = res

# # merge(num1, num2)
# # print(num1)


# class Btree:

#     class Node:

#         def __init__(self, value, left, right):

#             self.value = value
#             self.left = left
#             self.right = right

#         def __lt__(self, other):

#             return self.value < other.value

#         def __eq__(self, other):

#             return self.value == other.value
    
#     def __init__(self, *args):

#         self.__root = None
#         self.__length = 0

#         for x in args:

#             self.push(x)
    
#     def push(self, value):

#         if self.__root is None:

#             self.__root = self.Node(value, None, None)
        
#         else:

#             runner = self.__root

#             while runner is not None:

#                 if value < runner.value:

#                     if runner.left is None:

#                         runner.left = self.Node(value, None, None)
#                         break

#                     runner = runner.left
                
#                 else:

#                     if runner.right is None:

#                         runner.right = self.Node(value, None, None)
#                         break
                        
#                     runner = runner.right
        
#         self.__length += 1

#     def inorder(self):

#         if self.__root is None:

#             return None
        
#         values = []

#         def travelling(node):

#             if node is None:

#                 return
            
#             travelling(node.left)
#             travelling(node.right)
#             values.append(node.value)
        
#         travelling(self.__root)
#         return values
    
#     def __eq__(self, other):

#         if not isinstance(other, Btree):
#             return False
        
#         def travelling(node_self, node_other):

#             if node_self is None and node_other is None:
#                 return True
            
#             if node_self is None or node_other is None or node_self.value != node_other.value:
#                 return False

#             return (travelling(node_self.left, node_other.left) and 
#                     travelling(node_self.right, node_other.right))

#         return travelling(self.__root, other._Btree__root)
    
#     def height(self):
#         if self.__root is None:
#             return 0

#         def travelling(node):

#             if node is None:
#                 return 0
            
#             left_height = travelling(node.left)
#             right_height = travelling(node.right)
            
#             return 1 + max(left_height, right_height)

#         self.__height = travelling(self.__root)
#         return self.__height
    
#     def is_symmetric(self):
#         if not self.__root:
#             return True

#         queue = [self.__root]

#         while queue:

#             level_values = []
#             next_level_nodes = []
            
#             for node in queue:
#                 if node:
#                     level_values.append(node.value)
#                     next_level_nodes.append(node.left)
#                     next_level_nodes.append(node.right)
#                 else:
#                     level_values.append(None) # Важно для сохранения структуры

#             if level_values != level_values[::-1]:
#                 return False

#             # 3. Если на следующем уровне только None — мы закончили
#             if all(n is None for n in next_level_nodes):
#                 break
                
#             queue = next_level_nodes

#         return True
    
#     def LinkedList_to_BalancedTree(self, lst):

#         if not isinstance(lst, LinkedList):

#             return None

#         runner = lst.__Head
#         array = []
#         while runner is not None:

#             array.append(runner.value)
#             runner = runner.next
        
#         def building_tree(elements):

#             if not elements:

#                 return
            
#             mid = len(elements)//2

#             node = self.Node(elements[mid], None, None)

#             node.left = building_tree(elements[:mid])
#             node.right = building_tree(elements[mid+1:])

#             return node

#         self.__root = building_tree(array)
#         self.__length = len(array)
#         return self.__root



# lst = LinkedList(1,2,3,4,5,6)

class Graph:

    class Node:

        def __init__(self, name, value = float('inf')) -> None:

            self.name = name
            self.value = value
            self.neighbours = {}
    
    def __init__(self):

        self.__nodes = {}
    
    def add_node(self, name): # Здесь добавляю точку на карту

        if name not in self.__nodes:

            self.__nodes[name] = self.Node(name)
        
        return self.__nodes[name]
    
    def add_edge(self, name1: str, name2: str, weight: int): # Здесь создаю ребро между точками

        node1 = self.add_node(name1)
        node2 = self.add_node(name2)

        node1.neighbours[node2] = weight
        node2.neighbours[node1] = weight
    
    def Dijkstra(self, start_point):

        for node in self.__nodes.values(): # переинициализация весов в узлах

            node.value = float("inf")
        
        start_node = self.__nodes[start_point] # выбор стартовой точки
        start_node.value = 0 # так как у всех точек по умолчанию inf, то 0 заведомо меньше

        queue = list(self.__nodes.keys()) # очередь из все существующих вершин. Предполагается, что все они известны

        while queue: # пока все точки не будут просмотренны

            current_name = min(queue, key = lambda name: self.__nodes[name].value) # Выбираем точку, которая будет минимальной из всех
            current_node = self.__nodes[current_name] # точка, которую мы рассматриваем

            queue.remove(current_name) # Удаляем рассмотренную вершину

            if current_node.value == float("inf"):
                break

            for neighbor, weight in current_node.neighbours.items():

                distance = current_node.value + weight

                if distance < neighbor.value:
                    neighbor.value = distance
        
        return {name: node.value for name, node in self.__nodes.items()}

    def Dijkstra_bin(self, start_point):

        import heapq as hp

        for node in self.__nodes.values():

            node.value = float("inf")
        
        start_node = self.__nodes[start_point]
        start_node.value = 0

        priority_queue = [(0, start_point)]

        while priority_queue:

            current_distance, current_name = hp.heappop(priority_queue)
            current_node = self.__nodes[current_name]
            
            if current_distance > current_node.value:

                continue

            for neighbor_node, weight in current_node.neighbours.items():

                distance = current_node.value + weight
                
                if distance < neighbor_node.value:

                    neighbor_node.value = distance

                    hp.heappush(priority_queue, (distance, neighbor_node.name))
        return {name: node.value for name, node in self.__nodes.items()}

    def Dejkstra(self, start_point: str) -> dict: 

        """ Work principle """
        """1. It is considered that """

        for node in self.__nodes.values():

            node.value = float("inf")
        
        self.__nodes[start_point].value = 0

        queue = [self.__nodes[start_point]]

        while queue:

            current_node = min(queue, key = lambda x: x.value)

            queue.remove(current_node)

            for neighbour, weight in current_node.neighbours.items():

                distance = current_node.value + weight

                if distance < neighbour.value:

                    neighbour.value = distance

                    if neighbour not in queue:

                        queue.append(neighbour)
            
        return {name: node.value for name, node in self.__nodes.items()}

# numRows = int(input("Введите количество строк: "))

# def generate_pascal(n):

#     if n == 1:
#         return [[1]]
    
#     triangle = generate_pascal(n - 1)
#     prev_row = triangle[-1]
    
#     new_row = [1]
#     for i in range(len(prev_row) - 1):
#         new_row.append(prev_row[i] + prev_row[i+1])
#     new_row.append(1)
    
#     triangle.append(new_row)
#     return triangle

# if numRows > 0:
#     print(generate_pascal(numRows))
# else:
#     print([])
    


# num = str(input())

# if num[0] == "-":

#     num = num[1:]
#     sign = "-"
#     num = [sign] + [x for x in num[-1::-1]]

# else:

#     num = [x for x in num[-1::-1]]

# num = int("".join(num))

# print(num)


# s = str(input())
# numRows = int(input())

# def convert(s: str, numRows: int) -> str:

#     if numRows == 1 or numRows >= len(s):
#         return s
    
#     i = 0
#     flag = 0
#     mass = [[] for i in range(numRows)]

#     for c in s:

#         mass[i].append(c)

#         if i == numRows - 1:
#             flag = 1
#         if i == 0:
#             flag = 0

#         if not flag:
#             i += 1
#         else:
#             i -= 1
    
#     res = ""

#     for row in mass:

#         res += "".join(row)

#     return res

# print(convert(s, numRows))


# s = str(input())

# def strTonumATOI(s: str) -> int:
#     s = s.strip()
#     if not s:
#         return 0

#     sign = 1
#     start_idx = 0

#     # Обработка знака
#     if s[0] == '-':
#         sign = -1
#         start_idx = 1
#     elif s[0] == '+':
#         start_idx = 1

#     res_str = ""
#     for i in range(start_idx, len(s)):
#         if not s[i].isdigit():
#             break
#         res_str += s[i]

#     if not res_str:
#         return 0
    
#     num = int(res_str) * sign

#     INT_MIN, INT_MAX = -2**31, 2**31 - 1
#     if num < INT_MIN:
#         return INT_MIN
#     if num > INT_MAX:
#         return INT_MAX

#     return num

    
# print(strTonumATOI(s))

import torch

x = torch.randint(low = 1, high = 100, size = (10,4)).float()
mean = x.mean(dim=0)
std = x.std(dim=0)

x_std = (x-mean)/std
m = x.size(0)

# print(x_std)

x_wave = x_std.T @ x_std / (m-1)

# print(x_wave)

res = torch.linalg.eig(x_wave)

values = res.eigenvalues.real
vectors = res.eigenvectors.real
print(values, vectors, sep = "\n")

sorted_values, indices = torch.sort(values, descending=True)

sorted_vectors = vectors[:, indices]

print(f"Дисперсия: {sorted_values}")
print(f"Самый важный вектор {sorted_vectors[:, 0]}")

