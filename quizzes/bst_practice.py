class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST(object):
    def __init__(self, root):
        self.root = Node(root)

    def insert(self, new_val):
        new_node = Node(new_val)
        self._insert(self.root, new_node)
        
    def _insert(self, start, new_node):
        new_val = new_node.value
                
        if new_val < start.value:            
            if start.left:
                self._insert(start.left, new_node)
            else:                
                start.left = new_node
        else:
            if start.right:
                self._insert(start.right, new_node)
            else:
                start.right = new_node
            
    def search(self, find_val):
        return self._search(self.root, find_val)
    
    def _search(self, start, find_val):
        if start:
            if find_val == start.value:
                return True
            elif find_val < start.value:
                return self._search(start.left, find_val)            
            return self._search(start.right, find_val)
        return False
    
# Set up tree
tree = BST(4)

# Insert elements
tree.insert(2)
tree.insert(1)
tree.insert(3)
tree.insert(5)

# Check search
# Should be True
print tree.search(4)
# Should be False
print tree.search(6)