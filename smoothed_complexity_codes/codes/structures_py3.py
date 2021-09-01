'''
STV
Created on Dec 17, 2016
Updated on Jan 13, 2017
@author: Jun Wang
'''

from heapq import heappush, heappop

class Node:
    def __init__(self, value=None):
        self.value = value
        # self.length = length
        # self.children = children

    def __lt__(self, other):
        return 0


    def getvalue(self):
        return self.value

    def getchildren(self):
        return self.children

class Tree:
    def __init__(self, root=None):
        self.root = root

    def pre_order(self):
        if not self.root:
            return
        stackNode = []
        stackNode.append(self.root)
        while stackNode:
            node = stackNode.pop()
            if node.children:
                stackNode.append(node.children.reverse())


# class PriorityQueue:
#     def __init__(self):
#         self._queue = []
#
#     def put(self, item, priority):
#         myheappush(self._queue, (-priority, item))
#
#     def get(self):
#         return myheappop(self._queue)[-1]
#
#
# def myheappush(heap, item):
#     """Push item onto heap, maintaining the heap invariant."""
#     heap.append(item)
#     my_siftdown(heap, 0, len(heap) - 1)
#
#
# def myheappop(heap):
#     """Pop the smallest item off the heap, maintaining the heap invariant."""
#     lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
#     if heap:
#         returnitem = heap[0]
#         heap[0] = lastelt
#         my_siftup(heap, 0)
#         return returnitem
#     return lastelt
#
#
# def my_siftdown(heap, startpos, pos):
#     newitem = heap[pos]
#     # print(newitem[0])
#     # Follow the path to the root, moving parents down until finding a place
#     # newitem fits.
#     while pos > startpos:
#         parentpos = (pos - 1) >> 1
#         parent = heap[parentpos]
#         if newitem[0] < parent[0]:
#             heap[pos] = parent
#             pos = parentpos
#             continue
#         break
#     heap[pos] = newitem
#
#
# def my_siftup(heap, pos):
#     endpos = len(heap)
#     startpos = pos
#     newitem = heap[pos]
#     # Bubble up the smaller child until hitting a leaf.
#     childpos = 2*pos + 1    # leftmost child position
#     while childpos < endpos:
#         # Set childpos to index of smaller child.
#         rightpos = childpos + 1
#         if rightpos < endpos and not heap[childpos][0] < heap[rightpos][0]:
#             childpos = rightpos
#         # Move the smaller child up.
#         heap[pos] = heap[childpos]
#         pos = childpos
#         childpos = 2*pos + 1
#     # The leaf at pos is empty now.  Put newitem there, and bubble it up
#     # to its final resting place (by sifting its parents down).
#     heap[pos] = newitem
#     my_siftdown(heap, startpos, pos)


# search by HASH TABLE
def searchHash(hashtable, data):
    hashAddress = hash(data)
    while hashtable.get(hashAddress) and hashtable[hashAddress] != data:
        hashAddress += 1
        hashAddress = hash(hashAddress)
    if hashtable.get(hashAddress) is None:
        return None
    return hashAddress


# Insert data to HASH TABLE
def insertHash(hashtable, data):
    hashAddress=hash(data)
    # there is collision
    while(hashtable.get(hashAddress)):
        # open addressing
        hashAddress += 1
        hashAddress = hash(hashAddress)
    hashtable[hashAddress] = data


# path = 'D:\Social Choice\data\stv-m20n20-100k'
path = 'D:\Social Choice\data\m5n5-100-rp'
filename = '/Users/junwang/Documents/Social Choice/programming/SOC-M30N30-50k(0k-50k)-DFS.txt'