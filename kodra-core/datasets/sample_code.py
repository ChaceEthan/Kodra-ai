# Python & Algorithm Implementations Corpus for Kodra AI

SAMPLE_CODE_CORPUS = """
def kodra_ai_quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return kodra_ai_quick_sort(left) + middle + kodra_ai_quick_sort(right)

def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

class KodraNode:
    def __init__(self, value):
        self.value = value
        self.next = None

class KodraLinkedList:
    def __init__(self):
        self.head = None

    def append(self, value):
        new_node = KodraNode(value)
        if not self.head:
            self.head = new_node
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = new_node
"""
