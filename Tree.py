'''基础
144	Binary Tree Preorder Traversal	preorder
94	Binary Tree Inorder Traversal	Inorder
145	Binary Tree Postorder Traversal	postorder
102	Binary Tree Level Order Traversal	DFS + BFS
Preorder
100	Same Tree	preorder
101	Symmetric Tree	preorder
226	Invert Binary Tree	preorder + BFS
257	Binary Tree Paths	preorder
112	Path Sum	preorder
113	Path Sum II	preorder
129	Sum Root to Leaf Numbers	preorder
298	Binary Tree Longest Consecutive Sequence	preorder
111	Minimum Depth of Binary Tree	preorder
Postorder
104	Maximum Depth of Binary Tree	postorder
110	Balanced Binary Tree	postorder
124	Binary Tree Maximum Path Sum	postorder
250	Count Univalue Subtrees	postorder
366	Find Leaves of Binary Tree	postorder
337	House Robber III	postorder + preorder
BFS
107	Binary Tree Level Order Traversal II	BFS
103	Binary Tree Zigzag Level Order Traversal	BFS
199	Binary Tree Right Side View	BFS + preorder
BST
98	Validate Binary Search Tree	preorder
235	Lowest Common Ancestor of a Binary Search Tree	preorder
236	Lowest Common Ancestor of a Binary Tree	postorder
108	Convert Sorted Array to Binary Search Tree	binary search
109	Convert Sorted List to Binary Search Tree	binary search
173	Binary Search Tree Iterator	inorder
230	Kth Smallest Element in a BST	inorder
297	Serialize and Deserialize Binary Tree	BFS
285	Inorder Successor in BST	inorder
270	Closest Binary Search Tree Value	preorder
272	Closest Binary Search Tree Value II	inorder
99	Recover Binary Search Tree	inorder
'''

#144. Binary Tree Preorder Traversal
# Given a binary tree, return the preorder traversal of its nodes' values.
# Input: [1,null,2,3],Output: [1,2,3]
# Recursive solution is trivial, could you do it iteratively?
# iterative, stack LIFO

#recursive method
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root is None:
            return []
        self.ans =[]
        self.helper(root)
        return self.ans

    def helper(self, node):
        """
        Recurvise preorder traversal method
        :param node:
        :return:
        """
        # Visit the node first
        self.ans.append(node.val)
        # Then traverse left child
        if node.left:
            self.helper(node.left)
        # Traverse right child at last
        if node.right:
            self.helper(node.right)

# recursive method2
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        # recursive
        """
        if not root: return []
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)


# iterative method

    def preorderTraversal(self, root):
        ans = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                ans.append(node.val)
                stack.append(node.right)
                stack.append(node.left)
                #stack.extend([node.right, node.left])
        return ans


# 145.postorder
# recursive
class Solution:
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        # recursive
        """
        if not root: return []
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]

#iterative
from collections import namedtuple
Step = namedtuple('step', ['operation', 'node'])


class Solution:
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        results, stack = [], []
        stack.append(Step(0, root))
        while stack:
            current = stack.pop()
            if current.node:
                if current.operation == 1:
                    results.append(current.node.val)
                else:
                    stack.extend([Step(1, current.node), Step(0, current.node.right), Step(0, current.node.left)])
        return results

# iterative 2
class Solution:
    # @param {TreeNode} root
    # @return {integer[]}
    def postorderTraversal(self, root):
        ans, stack = [], [root]
        while stack:
            node = stack.pop()
            if node:
                # pre-order, right first
                ans.append(node.val)
                stack.append(node.left)
                stack.append(node.right)

        # reverse result
        return ans[::-1]


# 94.Inorder (in BST give sorted order)
# recursively
def inorderTraversal(self, root):
    res = []
    self.helper(root, res)
    return res


def helper(self, root, res):
    if root:
        self.helper(root.left, res)
        res.append(root.val)
        self.helper(root.right, res)


# iteratively
def inorderTraversal(self, root):
    res, stack = [], []
    while True:
        while root:
            stack.append(root)
            root = root.left
        if not stack:
            return res
        node = stack.pop()
        res.append(node.val)
        root = node.right

# iteratively2
from collections import namedtuple
Step = namedtuple('step', ['operation', 'node'])

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        # using stack
        """
        results, stack = [], []
        stack.append(Step(0, root))
        while stack:
            current = stack.pop()
            if current.node:
                if current.operation == 1:
                    results.append(current.node.val)
                else:
                    stack.extend([Step(0, current.node.right), Step(1, current.node), Step(0, current.node.left)])
        return results


#102. Binary Tree Level Order Traversal
# BFS (queue) )+ DFS (stack)
#Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).
#Given binary tree [3,9,20,null,null,15,7],return its level order traversal as:
# [
# #   [3],
# #   [9,20],
# #   [15,7]
# # ]
"""
queue的概念用deque来实现，popleft() 时间复杂为O(1)即可

外围的While用来定义BFS的终止条件，所以我们最开始initialize queue的时候可以直接把root放进去
在每层的时候，通过一个cur_level记录当前层的node.val，size用来记录queue的在增加子孙node之前大小，因为之后我们会实时更新queue的大小。
当每次从queue中pop出来的节点，把它的左右子节点放进Queue以后，记得把节点本身的的value放进cur_level
for loop终止后，就可以把记录好的整层的数值，放入我们的return数组里。
"""
from collections import deque

class Solution:
    def levelOrder(self, root):
        if not root: return []
        queue, res = deque([root]), []

        while queue:
            cur_level, size = [], len(queue)
            for i in range(size):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                cur_level.append(node.val)
            res.append(cur_level)
        return res
#method2 BFS with queue
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        ret = []
        queue = [root]
        next_lev = []
        while queue:
            th = []
            while queue:
                node = queue.pop(0)
                th.append(node.val)
                if node.left:
                    next_lev.append(node.left)
                if node.right:
                    next_lev.append(node.right)
            queue = next_lev
            next_lev = []
            ret.append(th)

        return ret

# other methods
#level is a list of the nodes in the current level. Keep appending a
# list of the values of these nodes to ans and then updating level
# with all the nodes in the next level (kids) until it reaches an empty
#  level. Python's list comprehension makes it easier to deal with many
# conditions in a concise manner.

#Solution 1, (6 lines)

def levelOrder(self, root):
    ans, level = [], [root]
    while root and level:
        ans.append([node.val for node in level])
        LRpair = [(node.left, node.right) for node in level]
        level = [leaf for LR in LRpair for leaf in LR if leaf]
    return ans

#Solution 2, (5 lines), same idea but use only one list comprehension in while loop to get the next level
def levelOrder(self, root):
    ans, level = [], [root]
    while root and level:
        ans.append([node.val for node in level])
        level = [kid for n in level for kid in (n.left, n.right) if kid]
    return ans

#Solution 3 (10 lines), just an expansion of solution 1&2 for better understanding.
def levelOrder(self, root):
    if not root:
        return []
    ans, level = [], [root]
    while level:
        ans.append([node.val for node in level])
        temp = []
        for node in level:
            temp.extend([node.left, node.right])
        level = [leaf for leaf in temp if leaf]
    return ans
#637 Average of Levels in binary tree



#100. same tree
#The "proper" way:

def isSameTree(self, p, q):
    if p and q:
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    return p is q
#The "tupleify" way:

def isSameTree(self, p, q):
    def t(n):
        return n and (n.val, t(n.left), t(n.right))
    return t(p) == t(q)

#other methods
def isSameTree1(self, p, q):
    if p and q:
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
    else:
        return p == q

# DFS with stack
def isSameTree2(self, p, q):
    stack = [(p, q)]
    while stack:
        node1, node2 = stack.pop()
        if not node1 and not node2:
            continue
        elif None in [node1, node2]:
            return False
        else:
            if node1.val != node2.val:
                return False
            stack.append((node1.right, node2.right))
            stack.append((node1.left, node2.left))
    return True

# BFS with queue
def isSameTree3(self, p, q):
    queue = [(p, q)]
    while queue:
        node1, node2 = queue.pop(0)
        if not node1 and not node2:
            continue
        elif None in [node1, node2]:
            return False
        else:
            if node1.val != node2.val:
                return False
            queue.append((node1.left, node2.left))
            queue.append((node1.right, node2.right))
    return True

#101. Symmetric Tree
def isSymmetric(self, root):
  if root is None:
      return True
  stack = [(root.left, root.right)]
  while stack:
      left, right = stack.pop()
      if left is None and right is None:
          continue
      if left is None or right is None:
          return False
      if left.val == right.val:
          stack.append((left.left, right.right))
          stack.append((left.right, right.left))
      else:
          return False
  return True

# iterative
from collections import deque


class Solution:
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        queue = deque()
        queue.extend([root, root])
        while queue:
            node1 = queue.popleft()
            node2 = queue.popleft()
            if not (node1 or node2):
                continue
            if not (node1 and node2):
                return False
            if node1.val != node2.val:
                return False
            queue.append(node1.left)
            queue.append(node2.right)
            queue.append(node2.right)
            queue.append(node1.left)
        return True


# recursive
class Solution:
    def _isMirror(self, node1, node2):
        if not (node1 or node2):
            return True
        if not (node1 and node2):
            return False
        return all([node1.val == node2.val, self._isMirror(node1.right, node2.left), self._isMirror(node2.right, node1.left)])

    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self._isMirror(root, root)


class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def helper(left, right):
            if not left and not right:
                return True
            if left and right:
                if left.val!=right.val:
                    return False
                else:
                    return helper(left.left, right.right) and helper(left.right, right.left)
            else:
                return False
        return helper(root,root)

#226	Invert Binary Tree	preorder + BFS
"""
Example:

Input:

     4
   /   \
  2     7
 / \   / \
1   3 6   9
Output:

     4
   /   \
  7     2
 / \   / \
9   6 3   1
"""

# recursive
def invertTree(self, root):
    if root:
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
#Maybe make it four lines for better readability:

def invertTree(self, root):
    if root:
        invert = self.invertTree
        root.left, root.right = invert(root.right), invert(root.left)
        return root
#And an iterative version using my own stack:

def invertTree(self, root):
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            node.left, node.right = node.right, node.left
            stack += node.left, node.right
    return root

#257	Binary Tree Paths	preorder
"""
Given a binary tree, return all root-to-leaf paths.

Note: A leaf is a node with no children.

Example:

Input:

   1
 /   \
2     3
 \
  5

Output: ["1->2->5", "1->3"]

Explanation: All root-to-leaf paths are: 1->2->5, 1->3
"""


# dfs + stack
def binaryTreePaths1(self, root):
    if not root:
        return []
    res, stack = [], [(root, "")]
    while stack:
        node, ls = stack.pop()
        if not node.left and not node.right:
            res.append(ls + str(node.val))
        if node.right:
            stack.append((node.right, ls + str(node.val) + "->"))
        if node.left:
            stack.append((node.left, ls + str(node.val) + "->"))
    return res


# bfs + queue
def binaryTreePaths2(self, root):
    if not root:
        return []
    res, queue = [], collections.deque([(root, "")])
    while queue:
        node, ls = queue.popleft()
        if not node.left and not node.right:
            res.append(ls + str(node.val))
        if node.left:
            queue.append((node.left, ls + str(node.val) + "->"))
        if node.right:
            queue.append((node.right, ls + str(node.val) + "->"))
    return res


# dfs recursively
def binaryTreePaths(self, root):
    if not root:
        return []
    res = []
    self.path(root, '', res)
    return res


def path(self, root, string, res):
    string += str(root.val)

    if root.left:
        self.path(root.left, string + '->', res)

    if root.right:
        self.path(root.right, string + '->', res)

    if not root.left and not root.right:
        res.append(string)


# faster solution
class Solution(object):
    def helper(self, root, path, ret):
        if root is None:
            return
        if root.left is None and root.right is None:
            ret.append(path+str(root.val))
        self.helper(root.left, path+str(root.val)+"->", ret)
        self.helper(root.right, path + str(root.val)+"->", ret)
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        ret = []
        self.helper(root, '', ret)
        return ret



#112	Path Sum	preorder
"""
Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.

Note: A leaf is a node with no children.

Example:

Given the below binary tree and sum = 22,

      5
     / \
    4   8
   /   / \
  11  13  4
 /  \      \
7    2      1
return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22."""

#recursion 的终止条件1 tree为空，2是leaf
class Solution(object):
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root is None:
            return False
        if root.left is None and root.right is None:
            if sum == root.val:
                return True
            return False
        return self.hasPathSum(root.left, sum - root.val) or \
               self.hasPathSum(root.right, sum - root.val)

class Solution:
    # @param root, a tree node
    # @param sum, an integer
    # @return a boolean
    # 1:27
    def hasPathSum(self, root, sum):
        if not root:
            return False

        if not root.left and not root.right and root.val == sum:
            return True

        sum -= root.val

        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)

# stack
class Solution(object):
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if not root:
            return False
        stack = []
        stack.append((root, sum-root.val))
        while stack:
            node, tSum = stack.pop()
            if not node.left and not node.right and tSum == 0:
                return True
            if node.left:
                stack.append((node.left, tSum-node.left.val))
            if node.right:
                stack.append((node.right, tSum-node.right.val))
        return False

# DFS Recursively
def hasPathSum1(self, root, sum):
    res = []
    self.dfs(root, sum, res)
    return any(res)


def dfs(self, root, target, res):
    if root:
        if not root.left and not root.right:
            if root.val == target:
                res.append(True)
        if root.left:
            self.dfs(root.left, target - root.val, res)
        if root.right:
            self.dfs(root.right, target - root.val, res)


# DFS with stack
def hasPathSum2(self, root, sum):
    if not root:
        return False
    stack = [(root, root.val)]
    while stack:
        curr, val = stack.pop()
        if not curr.left and not curr.right:
            if val == sum:
                return True
        if curr.right:
            stack.append((curr.right, val + curr.right.val))
        if curr.left:
            stack.append((curr.left, val + curr.left.val))
    return False


# BFS with queue
def hasPathSum(self, root, sum):
    if not root:
        return False
    queue = [(root, sum - root.val)]
    while queue:
        curr, val = queue.pop(0)
        if not curr.left and not curr.right:
            if val == 0:
                return True
        if curr.left:
            queue.append((curr.left, val - curr.left.val))
        if curr.right:
            queue.append((curr.right, val - curr.right.val))
    return False

#113	Path Sum II	preorder
"""
Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.

Note: A leaf is a node with no children.

Example:

Given the below binary tree and sum = 22,

      5
     / \
    4   8
   /   / \
  11  13  4
 /  \    / \
7    2  5   1
Return:

[
   [5,4,11,2],
   [5,8,4,5]
]
"""

# method3
def pathSum(self, root, sum):
    if not root:
        return []
    res = []
    self.dfs(root, sum, [], res)
    return res


def dfs(self, root, sum, ls, res):
    if not root.left and not root.right and sum == root.val:
        ls.append(root.val)
        res.append(ls)
    if root.left:
        self.dfs(root.left, sum - root.val, ls + [root.val], res)
    if root.right:
        self.dfs(root.right, sum - root.val, ls + [root.val], res)

# method2
def pathSum2(self, root, sum):
    if not root:
        return []
    if not root.left and not root.right and sum == root.val:
        return [[root.val]]
    tmp = self.pathSum(root.left, sum - root.val) + self.pathSum(root.right, sum - root.val)
    return [[root.val] + i for i in tmp]


# BFS + queue
def pathSum3(self, root, sum):
    if not root:
        return []
    res = []
    queue = [(root, root.val, [root.val])]
    while queue:
        curr, val, ls = queue.pop(0)
        if not curr.left and not curr.right and val == sum:
            res.append(ls)
        if curr.left:
            queue.append((curr.left, val + curr.left.val, ls + [curr.left.val]))
        if curr.right:
            queue.append((curr.right, val + curr.right.val, ls + [curr.right.val]))
    return res


# DFS + stack I
def pathSum4(self, root, sum):
    if not root:
        return []
    res = []
    stack = [(root, sum - root.val, [root.val])]
    while stack:
        curr, val, ls = stack.pop()
        if not curr.left and not curr.right and val == 0:
            res.append(ls)
        if curr.right:
            stack.append((curr.right, val - curr.right.val, ls + [curr.right.val]))
        if curr.left:
            stack.append((curr.left, val - curr.left.val, ls + [curr.left.val]))
    return res


# DFS + stack II
def pathSum5(self, root, s):
    if not root:
        return []
    res = []
    stack = [(root, [root.val])]
    while stack:
        curr, ls = stack.pop()
        if not curr.left and not curr.right and sum(ls) == s:
            res.append(ls)
        if curr.right:
            stack.append((curr.right, ls + [curr.right.val]))
        if curr.left:
            stack.append((curr.left, ls + [curr.left.val]))
    return res


#129	Sum Root to Leaf Numbers
"""
Input: [1,2,3]
    1
   / \
  2   3
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.
"""

#111	Minimum Depth of Binary Tree	preorder
"""
Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its minimum depth = 2.
"""
# DFS
def minDepth1(self, root):
    if not root:
        return 0
    if None in [root.left, root.right]:
        return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
    else:
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1

class Solution:
    # @param root, a tree node
    # @return an integer
    def minDepth(self, root):
        if root == None:
            return 0
        if root.left==None or root.right==None:
            return self.minDepth(root.left)+self.minDepth(root.right)+1

class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        l = self.minDepth(root.left)
        r = self.minDepth(root.right)
        if root.left is None:
            return r + 1
        elif root.right is None:
            return l + 1
        return min(l, r) + 1


# BFS
def minDepth(self, root):
    if not root:
        return 0
    queue = collections.deque([(root, 1)])
    while queue:
        node, level = queue.popleft()
        if node:
            if not node.left and not node.right:
                return level
            else:
                queue.append((node.left, level + 1))
                queue.append((node.right, level + 1))





#104	Maximum Depth of Binary Tree	postorder
"""
Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its depth = 3.
"""

def maxDepth(self, root):
    return 1 + max(map(self.maxDepth, (root.left, root.right))) if root else 0
#method2
class Solution:
    # @param {TreeNode} root
    # @return {integer}
    def maxDepth(self, root):
        if not root:
            return 0

        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        l = self.maxDepth(root.left)
        r = self.maxDepth(root.right)
        return max(l, r) + 1


#110	Balanced Binary Tree	postorder
# important data structure, gurantee worst situation as logN
"""Given a binary tree, determine if it is height-balanced.

For this problem, a height-balanced binary tree is defined as:

a binary tree in which the depth of the two subtrees of every 
node never differ by more than 1.

Example 1:

Given the following tree [3,9,20,null,null,15,7]:

    3
   / \
  9  20
    /  \
   15   7
Return true.

Example 2:

Given the following tree [1,2,2,3,3,null,null,4,4]:

       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
Return false.
"""
#time complexity: O(n)+2*O(n/2)+4*O(n/4)... = O(nlogn)
# recursive
class Solution(object):
    def isBalanced(self, root):

        def check(root):
            if root is None:
                return 0
            left = check(root.left)
            right = check(root.right)
            if left == -1 or right == -1 or abs(left - right) > 1:
                return -1
            return 1 + max(left, right)

        return check(root) != -1


def isBalanced(self, root: 'TreeNode') -> 'bool':
    self.balanced = True

    def dfs(node):
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        if not self.balanced or abs(left - right) > 1:
            self.balanced = False
        return max(left, right) + 1

    dfs(root)
    return self.balanced

#有了104的基础，我们在延伸下看看110这道题，其实就是基于高度计算，然后判断一下。
#但由于嵌套的Recursion调用，整体的时间复杂度是：O(nlogn) , 在每一层调用get_height的平均时间复杂度是O(N)，
# 然后基于二叉树的性质，调用了的高度是logn，所以n * logn 的时间复杂。

class Solution(object):
    def isBalanced(self, root):
        if not root: return True
        left = self.get_height(root.left)
        right = self.get_height(root.right)
        if abs(left - right) > 1:
            return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)

    def get_height(self, root):
        if not root: return 0
        left = self.get_height(root.left)
        right = self.get_height(root.right)
        return max(left, right) + 1


#上面这种Brute Froce的方法，整棵树有很多冗余无意义的遍历，其实我们在处理完get_height这个高度的时候，
# 我们完全可以在检查每个节点高度并且返回的同时，记录左右差是否已经超过1，只要有一个节点超过1，
# 那么直接返回False即可，因此我们只需要在外围设立一个全球变量记录True和False，在调用get_height的时候，
# 内置代码里加入对左右高度的判定即可，代码如下

#时间复杂度: O(N)
# Recursive Rules:
# 索取：Node的左孩子是不是全部是Balanced，Node的右孩子是不是全部是Balanced的，
# 返回：如果都是Balanced的，返回True，不然返回False

class Solution(object):
    def isBalanced(self, root):
        self.flag = True
        self.getHeight(root)
        return not self.flag

    def getHeight(self, root):
        if not root: return 0
        left = self.getHeight(root.left)
        right = self.getHeight(root.right)
        if abs(left - right) > 1:
            self.flag = False
        return max(left, right) + 1


#最后Leetcode上有一种 - 1的方法，其实就是上面这种方法的一种延伸。如果左右两边出现了高度差高于1的情况，直接返回 - 1，这个 - 1
#怎么来的？因为高度不可能为负数，-1 其实就是一种True / False的表达。

#那么在实现上，我们只要对get_height每次返回前做一个判定即可，具体实现看下方：

#时间复杂度: O(N)
class Solution(object):
    def isBalanced(self, root):
        height = self.get_height(root)
        return height != -1

    def get_height(self, root):
        if not root: return 0
        left = self.get_height(root.left)
        right = self.get_height(root.right)
        if left == -1 or right == -1: return -1
        if abs(left - right) > 1:  return -1
        return max(left, right) + 1


#Iterative, based on postorder traversal:
class Solution(object):
    def isBalanced(self, root):
        stack, node, last, depths = [], root, None, {}
        while stack or node:
            if node:
                stack.append(node)
                node = node.left
            else:
                node = stack[-1]
                if not node.right or last == node.right:
                    node = stack.pop()
                    left, right = depths.get(node.left, 0), depths.get(node.right, 0)
                    if abs(left - right) > 1: return False
                    depths[node] = 1 + max(left, right)
                    last = node
                    node = None
                else:
                    node = node.right
        return True

#124	Binary Tree Maximum Path Sum	postorder
# https://www.youtube.com/watch?v=9ZNky1wqNUw
"""
Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes from some 
starting node to any node in the tree along the parent-child connections. 
The path must contain at least one node and does not need to go through 
the root.

Example 1:

Input: [1,2,3]

       1
      / \
     2   3

Output: 6
Example 2:

Input: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

Output: 42"""
# 花花酱
class Solution(object):
    def _maxPathSum(self, root):
        if not root: return -sys.maxint
        l = max(0, self._maxPathSum(root.left))
        r = max(0, self._maxPathSum(root.right))
        self.ans= max(self.ans, root.val + l + r)
        return root.val + max(l, r)

    def maxPathSum(self, root):
        self.ans = -sys.maxint
        self._maxPathSum(root)
        return self.ans
# method2
class Solution(object):
    current_max = float('-inf')
    def maxPathSum(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
        self.maxPathSumHelper(root)
        return self.current_max

    def maxPathSumHelper(self, root):
        """Helper method"""
        if root is None:
            return root
        left = self.maxPathSumHelper(root.left)
        right = self.maxPathSumHelper(root.right)
        left = 0 if left is None else (left if left > 0 else 0)
        right = 0 if right is None else (right if right > 0 else 0)
        self.current_max = max(left+right+root.val, self.current_max)
        return max(left, right) + root.val

#Solution 1: Helper returning two values: (240 ms, 8 lines)

def maxPathSum(self, root):
    def maxsums(node):
        if not node:
            return [-2**31] * 2
        left = maxsums(node.left)
        right = maxsums(node.right)
        return [node.val + max(left[0], right[0], 0),
                max(left + right + [node.val + left[0] + right[0]])]
    return max(maxsums(root))
#My helper function returns two values:

#The max sum of all paths ending in the given node (can be extended through the parent)
#The max sum of all paths anywhere in tree rooted at the given node (can not be extended through the parent).
#Solution 2: Helper updating a "global" maximum: (172 ms, 10 lines)

def maxPathSum(self, root):
    def maxend(node):
        if not node:
            return 0
        left = maxend(node.left)
        right = maxend(node.right)
        self.max = max(self.max, left + node.val + right)
        return max(node.val + max(left, right), 0)
    self.max = None
    maxend(root)
    return self.max
#Here the helper is similar, but only returns the first of the two values (the max sum of all paths ending in the
# given node). Instead of returning the second value (the max sum of all paths anywhere in tree rooted at the given
# node), it updates a "global" maximum.


#543 Diameter of Binary Tree
"""
Given a binary tree, you need to compute the length of the diameter of the tree. 
The diameter of a binary tree is the length of the longest path between any two nodes 
in a tree. This path may or may not pass through the root.

Example:
Given a binary tree 
          1
         / \
        2   3
       / \     
      4   5    
Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].

Note: The length of path between two nodes is represented by the number of edges 
between them."""

#Let's calculate the depth of a node in the usual way: max(depth of node.left, depth of node.right) + 1.
#  While we do, a path "through" this node uses 1 + (depth of node.left) + (depth of node.right) nodes.
# Let's search each node and remember the highest number of nodes used in some path. The desired length
# is 1 minus this number.
def diameterOfBinaryTree(self, root):
    self.ans = 1

    def depth(root):
        if not root: return 0
        ansL = depth(root.left)
        ansR = depth(root.right)
        self.ans = max(self.best, ansL + ansR + 1)
        return 1 + max(ansL, ansR)

    depth(root)
    return self.ans - 1



#687 Longest Univalue Path
"""
Given a binary tree, find the length of the longest path where each node 
in the path has the same value. This path may or may not pass through the 
root.

Note: The length of path between two nodes is represented by the number 
of edges between them.

Example 1:

Input:

              5
             / \
            4   5
           / \   \
          1   1   5
Output:

2"""


class Solution(object):
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.longest = 0
        self.dfs(root)
        return self.longest

    def dfs(self, node):
        if not node:
            return None, 0
        if not node.left and not node.right:
            return node.val, 1
        left, countl = self.dfs(node.left)
        right, countr = self.dfs(node.right)
        if left == node.val and right == node.val:
            self.longest = max(self.longest, countl + countr)
            return node.val, max(countl, countr) + 1
        if left == node.val:
            self.longest = max(self.longest, countl)
            return node.val, countl + 1
        if right == node.val:
            self.longest = max(self.longest, countr)
            return node.val, countr + 1

        return node.val, 1


class Solution(object):
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def postOrder(node, v):
            if not node:
                return 0
            l = postOrder(node.left, node.val)
            r = postOrder(node.right, node.val)
            if l + r > self.max_length:
                self.max_length = l + r
            if node.val != v:
                return 0
            else:
                return max(l, r) + 1

        self.max_length = 0
        postOrder(root, None)
        return self.max_length



class Solution(object):
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # Time: O(n)
        # Space: O(n)
        longest = [0]       # 不理解为什么这里是list 而不是直接longest =0， 但是改成0 就跑不通了, 要用 self.longest = 0
        def traverse(node):
            if not node:
                return 0
            left_len, right_len = traverse(node.left), traverse(node.right)
            left = (left_len + 1) if node.left and node.left.val == node.val else 0
            right = (right_len + 1) if node.right and node.right.val == node.val else 0
            longest[0] = max(longest[0], left + right)
            return max(left, right)
        traverse(root)
        return longest[0]


class Solution(object):
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        self.longest = 0

        def traverse(node):
            if not node:
                return 0
            left_len, right_len = traverse(node.left), traverse(node.right)
            if node.left and node.val == node.left.val:
                left = left_len + 1
            else:
                left = 0
            if node.right and node.val == node.right.val:
                right = right_len + 1
            else:
                right = 0
            self.longest = max(self.longest, left + right)
            return max(left, right)

        traverse(root)
        return self.longest


class Solution(object):
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        # recursive

        if root == None:
            return 0

        if root.left == None and root.right == None:
            return 0

        self.ans = 0

        def longest(node):
            if not node: return 0
            left = longest(node.left)
            right = longest(node.right)
            a = 0
            b = 0
            if node.left and node.left.val == node.val:
                a = left + 1
            if node.right and node.right.val == node.val:
                b = right + 1
            self.ans = max(a + b, self.ans)
            return max(a, b)

        longest(root)
        return self.ans


# locked 250	Count Univalue Subtrees	postorder

# locked 366	Find Leaves of Binary Tree	postorder


# 337	House Robber III	postorder + preorder

class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def superrob(node):
            # returns tuple of size two (now, later)
            # now: max money earned if input node is robbed
            # later: max money earned if input node is not robbed

            # base case
            if not node: return (0, 0)

            # get values
            left, right = superrob(node.left), superrob(node.right)

            # rob now
            now = node.val + left[1] + right[1]

            # rob later
            later = max(left) + max(right)

            return (now, later)

        return max(superrob(root))
# method2
class Solution(object):
    def rob(self, root):
        return self.robDFS(root)[1];
    def robDFS(self,node):
        if node is None:
            return (0,0)
        l = self.robDFS(node.left)
        r = self.robDFS(node.right)
        return (l[1] + r[1], max(l[1] + r[1], l[0] + r[0] + node.val))


# 938
class Solution:
    def rangeSumBST(self, root, L, R):
        if not root: return 0
        l = self.rangeSumBST(root.left, L, R)
        r = self.rangeSumBST(root.right, L, R)
        return l + r + (L <= root.val <= R) * root.val

class Solution:
    def rangeSumBST(self, root: TreeNode, L: int, R: int) -> int:
        if not root:
            return 0
        elif root.val >= L and root.val <= R:
            return root.val + self.rangeSumBST(root.left, L, R) + self.rangeSumBST(root.right, L, R)
        elif root.val < L:
            return self.rangeSumBST(root.right, L, R)
        else:
            return self.rangeSumBST(root.left, L, R)

class Solution:
    def rangeSumBST(self, root, L, R):
        sum = 0
        if root is None:
            return sum
        if L<= root.val <= R:
            sum += root.val

        if root.val > L:
            sum += self.rangeSumBST(root.left, L, R)
        if root.val < R:
            sum += self.rangeSumBST(root.right, L, R)

        return sum

# stack structure
class Solution:
    def rangeSumBST(self, root, L, R):
        sum = 0
        stack = [root]
        while stack:
            node = stack.pop()

        if L <= node.val <= R:
            sum += node.val

        if node.val > L and node.left is not None:
            stack.append(node.left)
        if node.val < R and node.right is not None:
            stack.append(node.right)

        return sum


# fast example
class Solution(object):
    def rangeSumBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: int
        """
        self.result = 0

        def traversal(node):
        	if node is None:
        		return
        	if node.left is not None and node.val >= L:
        		# if too small, no need to traverse left
        		traversal(node.left)
        	if node.val >= L and node.val <= R:
        		self.result += node.val
        	if node.right is not None and node.val <= R:
        		traversal(node.right)
        traversal(root)
        return self.result






# 617

class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if t1 is None:
            return t2
        if t2 is None:
            return t1
        t1.val += t2.val
        t1.left = self.mergeTrees(t1.left, t2.left)
        t1.right = self.mergeTrees(t1.right, t2.right)
        return t1

class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """

        if t1 == None:
            return t2
        if t2 == None:
            return t1

        t1.val += t2.val

        t1.left = self.mergeTrees(t1.left, t2.left)
        t1.right = self.mergeTrees(t1.right, t2.right)

        return t1



def mergeTrees(self, t1, t2):
    if not t1 and not t2: return None
    ans = TreeNode((t1.val if t1 else 0) + (t2.val if t2 else 0))
    ans.left = self.mergeTrees(t1 and t1.left, t2 and t2.left)
    ans.right = self.mergeTrees(t1 and t1.right, t2 and t2.right)
    return ans

# 700
class Solution(object):
    def searchBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        if not root:
            return null

        while root:
            if root.val == val:
                return root
            elif root.val > val:
                root = root.left
            else:
                root = root.right


class Solution(object):
    def searchBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        if root:
            if root.val == val:
                return root
            elif root.val < val:
                return self.searchBST(root.right,val)
            else:
                return self.searchBST(root.left,val)
        return None

# 589. N-ary Tree Preorder Traversal
# recursive
class Solution(object):
    def preorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        if not root:
            return []
        traversal = [root.val]
        for child in root.children:
            traversal.extend(self.preorder(child))
        return traversal

# iterative
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children
"""
class Solution(object):
    def preorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        if not root:
            return []
        traversal = []
        stack = [root]
        while stack:
            cur = stack.pop()
            traversal.append(cur.val)
            stack.extend(reversed(cur.children))
        return traversal

# 590
class Solution(object):
    def postorder(self, root):
        ret, stack = [], root and [root]
        while stack:
            node = stack.pop()
            ret.append(node.val)
            stack += [child for child in node.children if child]
        return ret[::-1]

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children


"""
class Solution(object):
    def postorder(self, root):
        """
        :type
        root: Node
        :rtype: List[int]
        """
        if not root:
            return []

        stack, output = [root], []
        while stack:
            root = stack.pop()
            if root:
                output.append(root.val)
                for c in root.children:
                    stack.append(c)
        return output[::-1]

# 965
class Solution:
    def isUnivalTree(self, root):
        if not root:
            return True

        if root.right:
            if root.val != root.right.val:  # Equavalent
                return False

        if root.left:
            if root.val != root.left.val:  # Equavalent
                return False

        return self.isUnivalTree(root.right) and self.isUnivalTree(root.left)

# 559. Maximum Depth of N-ary Tree
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Node
        :rtype: int
        """
        if not root:   #if root is None:
            return 0
        depth = 0

        for child in root.children:
            depth = max(depth, self.maxDepth(child))
        return depth + 1


# 897. Increasing Order Search Tree
    """
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
    """

"""
tail is its next node in inorder,
（the word next may be easier to understand, but it’s a keyword in python)
https://leetcode.com/problems/increasing-order-search-tree/discuss/165885/C%2B%2BJavaPython-Self-Explained-5-line-O(N)
"""
def increasingBST(self, root, tail=None):   # 自己加了个tail？
    if root is None:
        return tail
    x = TreeNode(root.val)
    x.right = self.increasingBST(root.right, tail)
    return self.increasingBST(root.left, x)

# 872. Leaf-Similar Trees
class Solution(object):
    def leafSimilar(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: bool
        """
        return self.findleaf(root1) == self.findleaf(root2)

    def findleaf(self, root):
        if not root: return []
        if not (root.left or root.right): return [root.val]
        return self.findleaf(root.left) + self.findleaf(root.right)

# 429. N-ary Tree Level Order Traversal
class Solution(object):
    def levelOrder(self, root):
        q, ret = [root], []
        while any(q):
            ret.append([node.val for node in q])
            q = [child for node in q for child in node.children]
        return ret

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: Node
        :rtype: List[List[int]]
        """
        if not root:return []
        res = []
        stack = [root]
        while stack:
            temp = []
            next_stack = []
            for node in stack:
                temp.append(node.val)
                for child in node.children:
                    next_stack.append(child)
            stack = next_stack
            res.append(temp)
        return res


class Solution(object):
    def levelOrder(self, root):
        """
        :type root: Node
        :rtype: List[List[int]]
        """
        self.ans = []

        def DFS(root, depth):
            if root is None:
                return
            while len(self.ans) <= depth:
                self.ans.append([])
            self.ans[depth].append(root.val)
            for child in root.children:
                DFS(child, depth + 1)

        DFS(root, 0)
        return self.ans


"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children
"""


class Solution(object):
    def levelOrder(self, root):
        """
        :type root: Node
        :rtype: List[List[int]]
        """
        ret = []

        if not root:
            return ret

        queue = [(root, 0)]
        tmp = []
        cur = 0
        while queue:
            node, level = queue.pop(0)
            for child in node.children:
                queue.append((child, level + 1))
            if level != cur:
                ret.append(tmp)
                tmp = []
                cur += 1
            tmp.append(node.val)

        ret.append(tmp)
        return ret

# 1022. Sum of Root To Leaf Binary Numbers
class Solution(object):
    def sumRootToLeaf(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        self.path = ''
        self.count = 0

        def traverse(node, path):
            if not node:
                return 0

            if node.left:
                traverse(node.left, path + str(node.val))

            if node.right:
                traverse(node.right, path + str(node.val))

            if not node.left and not node.right:
                path += str(node.val)
                self.count += int(path, 2)

        traverse(root, '')
        return self.count


# method 1
    # it's quite similar to find and return all the root-to-leaf paths
    # But this time we just need to return the decimal
	# O(n) for time
	# and without taking the recursion space into account
	# we will have O(n) for additional space
class Solution(object):
    def sumRootToLeaf(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return dfs(root)

        def dfs(node, path=None):
            if path == None:
                path = ''
            if node:
                path += str(node.val)
                if node.left or node.right:
                    return dfs(node.left, path) + dfs(node.right, path)
                else:
                    return int(path, 2)
            else:
                return 0



#method 2 (recommended)
    # very similar to the implementation of method 1
	# but this time, we directly pass the parent sum instead of
	# only calculate the decimal presentation in the leaf
	# O(n) for time
	# and without taking the recursion space into account
	# we will have O(1) for additional space
    def dfs2(node, parent_sum=None):
        if parent_sum == None:
            parent_sum = 0
        if node:
            parent_sum = parent_sum * 2 + node.val
            if node.left or node.right:
                return dfs2(node.left, parent_sum) + dfs2(node.right, parent_sum)
            else:
                return parent_sum
        else:
            return 0


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumRootToLeaf(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        self.total = 0

        def rec(node, cbuf=""):
            if not node:
                return
            if not node.left and not node.right:
                cval = cbuf + str(node.val)
                self.total += int(cval, 2)
            else:
                tmp = cbuf + str(node.val)
                rec(node.left, tmp)
                rec(node.right, tmp)

        rec(root)
        return self.total % (10 ** 9 + 7)

# 637. Average of Levels in Binary Tree
class Solution:
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        if root is None:
            return []

        result = []
        current_level = [root]
        while current_level:
            level_nodes = []
            next_level = []

            for node in current_level:
                level_nodes.append(node.val)
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)

            result.append(sum(level_nodes) / float(len(level_nodes)))
            current_level = next_level
        return result

class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            avg = 0.0
            newq = []
            for q in queue:
                if q.left:
                    newq.append(q.left)
                if q.right:
                    newq.append(q.right)
                avg += q.val
            res.append(avg/len(queue))
            queue = newq
        return res

# 653. Two Sum IV - Input is a BST
    def findTarget(self, root, k):
        if not root: return False
        bfs, s = [root], set()
        for i in bfs:
            if k - i.val in s: return True
            s.add(i.val)
            if i.left: bfs.append(i.left)
            if i.right: bfs.append(i.right)
        return False

# 993 Cousins in Binary Tree

class Solution(object):
    def isCousins(self, root, x, y):
        """
        :type root: TreeNode
        :type x: int
        :type y: int
        :rtype: bool
        """

        def BFT(node, level):
            if node:
                depth[node.val] = level
                if node.left:
                    parent[node.left.val] = node
                    BFT(node.left, level + 1)
                if node.right:
                    parent[node.right.val] = node
                    BFT(node.right, level + 1)

        depth = {}
        parent = {root: None}
        BFT(root, 0)
        if depth[x] == depth[y] and parent[x] != parent[y]:
            return True
        else:
            return False



class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        def dfs(node, parent, depth, mod):
            if node:
                if node.val == mod:
                    return depth, parent
                return dfs(node.left, node, depth + 1, mod) or dfs(node.right, node, depth + 1, mod)
        dx, px, dy, py = dfs(root, None, 0, x) + dfs(root, None, 0, y)
        return dx == dy and px != py


class Solution(object):
    def isCousins(self, root, x, y):
        """
        :type root: TreeNode
        :type x: int
        :type y: int
        :rtype: bool
        """
        lookup = {}
        def dfs(root, i=0, p=None):
            if root:
                if root.val in (x, y): lookup[root.val] = (i, p)
                dfs(root.left, i=i+1, p=root.val)
                dfs(root.right, i=i+1, p=root.val)
        dfs(root)
        return lookup[x][0] == lookup[y][0] and lookup[x][1] != lookup[y][1]


# 108. Convert Sorted Array to Binary Search Tree

class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        def convert(left, right):
            if left > right:
                return None
            mid = (left + right) // 2
            node = TreeNode(nums[mid])
            node.left = convert(left, mid - 1)
            node.right = convert(mid + 1, right)
            return node
        return convert(0, len(nums) - 1)


class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        # node = TreeNode(None)
        # if not nums:
        #     return None
        # node_index = len(nums)//2
        # node.val = nums[node_index]
        # node.left = self.sortedArrayToBST(nums[:node_index])
        # node.right = self.sortedArrayToBST(nums[node_index+1:])
        # return node
        def built(start,end):
            if start >= end:
                return None
            mid = (start + end) //2
            node = TreeNode(nums[mid])
            node.left = built(start,mid)
            node.right = built(mid+1,end)
            return node
        return built(0,len(nums))


class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return
        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid + 1:])

        return root



# 606. Construct String from Binary Tree
class Solution:
    def tree2str(self, t: TreeNode) -> str:
        if not t:
            return ''
        if not t.right and not t.left:
            return str(t.val)
        if not t.right:
            return str(t.val) + '(' + self.tree2str(t.left) + ')'
        if not t.left:
            return str(t.val) + '()' + '(' + self.tree2str(t.right) + ')'

        return str(t.val) + '(' + self.tree2str(t.left) + ')' + '(' + self.tree2str(t.right) + ')'

class Solution(object):
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        if not t: return ("")
        l, r = "", ""
        if t.left:
            l = "({})".format(self.tree2str(t.left))
        if t.right:
            r = "({})".format(self.tree2str(t.right))
        if (not t.left) and len(r) > 0:              # this is important
            l = "()"
        return (str(t.val) + l + r)


class Solution(object):
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        def tree2str(root, res):
            if not root: return
            res.append(str(root.val))
            if not root.left and not root.right:
                res.append(")")
                return
            if not root.left:
                res.append("()")
            else:
                res.append("(")
                tree2str(root.left, res)
            if root.right:
                res.append("(")
                tree2str(root.right, res)
            res.append(")")
        res = []
        tree2str(t, res)
        return "".join(res[:-1])


class Solution(object):
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """

        def dfs(node, par):
            if not node:
                if par and par.right:
                    return '()'
                else:
                    return ''
            return '(' + str(node.val) + dfs(node.left, node) + dfs(node.right, node) + ')'

        return dfs(t, None)[1:-1]


 def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        def preorder(root):
            if root is None:
                return ""
            s=str(root.val)
            l=preorder(root.left)
            r=preorder(root.right)
            if r=="" and l=="":
                return s
            elif l=="":
                s+="()"+"("+r+")"
            elif r=="":
                s+="("+l+")"
            else :
                s+="("+l+")"+"("+r+")"
            return s
        return preorder(t)


# 538. Convert BST to Greater Tree
class Solution(object):

    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        self.s = 0
        self.revInorder(root)
        return root

    def revInorder(self, root):
        if root is None:
            return 0

        self.revInorder(root.right)
        self.s += root.val
        root.val = self.s
        self.revInorder(root.left)
        return root

class Solution(object):
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        cursor = root
        stack = []
        val = 0
        while cursor or stack:
            while cursor:
                stack.append(cursor)
                cursor = cursor.right
            cursor = stack.pop()
            cursor.val += val
            val = cursor.val
            cursor = cursor.left
        return root


class Solution(object):
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        def f(root,val):
            if root is None:
                return val
            root.val+=f(root.right,val)
            return f(root.left,root.val)
        f(root,0)
        return root


class Solution(object):
    def convertBST(self, root):
        total = 0
        node = root
        stack = []
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.right
            node = stack.pop()
            total += node.val
            node.val = total
            node = node.left
        return root

# 530. Minimum Absolute Difference in BST
def getMinimumDifference(self, root):
    L = []

    def dfs(node):
        if node.left: dfs(node.left)
        L.append(node.val)
        if node.right: dfs(node.right)

    dfs(root)
    return min(abs(a - b) for a, b in zip(L, L[1:]))



class Solution(object):
    def getMinimumDifference(self, root):
        stack =[]
        diff = float('inf')
        prev = float('inf')
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            diff = min(diff, abs(root.val-prev))
            prev = root.val
            root = root.right
        return diff

# 783. Minimum Distance Between BST Nodes
# same as above?
class Solution(object):
    def minDiffInBST(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        L = []

        def dfs(node):
            if node.left: dfs(node.left)
            L.append(node.val)
            if node.right: dfs(node.right)

        dfs(root)
        return min(abs(a - b) for a, b in zip(L, L[1:]))


class Solution(object):
    def __init__(self):
        self.minDiff = float("inf")
        self.last = -float("inf")

    def minDiffInBST(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return float("inf"), float("inf")
        self.minDiffInBST(root.left)
        self.minDiff = min(self.minDiff, root.val - self.last)
        self.last = root.val
        self.minDiffInBST(root.right)
        return self.minDiff


class Solution(object):
    a, b = float("-inf"), float("inf")

    def minDiffInBST(self, r):
        if not r: return r
        self.minDiffInBST(r.left)
        self.b, self.a = min(self.b, r.val - self.a), r.val
        self.minDiffInBST(r.right)
        return self.b


# 404. Sum of Left Leaves
class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0

        self.res = 0
        queue = [(root, "r")]
        while queue:
            (node, direction) = queue.pop(0)
            left, right = (node.left, "l"), (node.right, "r")
            if left[0] is None:
                if direction == "l" and right[0] is None:
                    self.res += node.val
                    # print(self.res)
            else:
                queue.append(left)
            if right[0] is not None:
                queue.append(right)
        return self.res


class Solution(object):
    def sumOfLeftLeaves(self, root):
        if not root:
            return 0
        ssum = 0
        curr = [root]
        while curr:
            nxt = []
            for node in curr:
                if node.left:
                    if not node.left.left and not node.left.right:
                        ssum += node.left.val
                    nxt.append(node.left)
                if node.right:
                    nxt.append(node.right)
            curr = nxt
        return ssum

class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        if root.left and not root.left.left and not root.left.right:
            return root.left.val + self.sumOfLeftLeaves(root.right)
        return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)


class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def dfs(root):
            if root is None:
                return 0
            if root.left and root.left.left is None and root.left.right is None:
                print(root.left.val)
                return root.left.val + dfs(root.left) + dfs(root.right)
            else:
                return dfs(root.left) + dfs(root.right)

        return dfs(root)


# 107. Binary Tree Level Order Traversal II
"""Given a binary tree, return the bottom-up level order traversal of its nodes' values. 
(ie, from left to right, level by level from leaf to root).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its bottom-up level order traversal as:
[
  [15,7],
  [9,20],
  [3]
]
"""
# method 1

class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        from Queue import *
        queue1 = Queue()
        res = []
        queue1.put(root)

        while not queue1.empty():
            curr_level = []
            size1 = queue1.qsize()
            for i in range(size1):
                curr_node = queue1.get()
                if curr_node.left:
                    queue1.put(curr_node.left)
                if curr_node.right:
                    queue1.put(curr_node.right)
                curr_level.append(curr_node.val)
            res.append(curr_level)
        return res[::-1]

# method 2

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

def find_max(node, depth):
    if node == None:
        return depth

    depth += 1
    l = find_max(node.left, depth)
    r = find_max(node.right, depth)
    return max(l, r)


def put_level(node, depth, m, res):
    if node == None:
        return
    depth += 1
    res[m - depth].append(node.val)
    put_level(node.left, depth, m, res)
    put_level(node.right, depth, m, res)


class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        m = find_max(root, 0)
        res = [[] for i in range(m)]
        put_level(root, 0, m, res)
        return res

# method 3
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """

        if not root:
            return []

        stack = [root]

        result = []

        while True:
            temp = []

            nodes = []
            while stack:
                node = stack.pop(0)

                nodes.append(node.val)

                if node.left:
                    temp.append(node.left)

                if node.right:
                    temp.append(node.right)

            result.append(nodes)

            if not temp:
                return result[::-1]

            stack = temp

# method 4
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """

        res = []
        self.dfs(root, 0, res)
        return res

    def dfs(self, root, level, res):
        if root:
            if len(res) < level + 1:
                res.insert(0, [])
            res[-(level + 1)].append(root.val)
            self.dfs(root.left, level + 1, res)
            self.dfs(root.right, level + 1, res)


# 563 . Binary Tree Tilt
class Solution(object):
    def findTilt(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0

        self.ans = 0

        # Returns sums of all node values, including itself
        # Calculates answer as side effect
        def helper(node):
            if not node:
                return 0
            sum_left = helper(node.left)
            sum_right = helper(node.right)

            tilt = sum_left - sum_right
            if tilt < 0:
                tilt = -tilt
            self.ans += tilt

            return node.val + sum_left + sum_right

        helper(root)

        return self.ans

# 235. Lowest Common Ancestor of a Binary Search Tree
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        elif root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:   # much faster without else here
            return root

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        node = root
        while node:
            if node.val < p.val and node.val < q.val:
                node = node.right
            elif node.val > p.val and node.val > q.val:
                node = node.left
            else:
                return node


class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None

        if root == p or root == q:
            return root

        if root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        return root

def lowestCommonAncestor(self, root, p, q):
    while (root.val - p.val) * (root.val - q.val) > 0:
        root = (root.left, root.right)[p.val > root.val]
    return root

def lowestCommonAncestor(self, root, p, q):
    while root:
        v, pv, qv = root.val, p.val, q.val
        if v > max(pv, qv): root = root.left
        elif v < min(pv, qv): root = root.right
        else: return root

# 437. Path Sum III
"""You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).

The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.

Example:

root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
"""
from collections import defaultdict


class Solution(object):

    def pathSum(self, root, target):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        self.paths = 0

        cache = defaultdict(int)
        cache[0] = 1

        def dfs(node, currentSum):
            if node == None:
                return

            currentSum += node.val
            findOld = currentSum - target
            self.paths += cache[findOld]

            cache[currentSum] += 1

            dfs(node.left, currentSum)
            dfs(node.right, currentSum)

            cache[currentSum] -= 1

        dfs(root, 0)

        return self.paths


class Solution(object):
    def pathSum(self, root, s):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        if not root:
            return 0
        queue = [(root, [root.val])]
        cnt = 0

        while queue:
            node, val_list = queue.pop(0)
            cnt += val_list.count(s)

            if node.left:
                queue.append((node.left, [val + node.left.val for val in val_list] + [node.left.val]))
            if node.right:
                queue.append((node.right, [val + node.right.val for val in val_list] + [node.right.val]))

        return cnt

 # 两个recursion： 一个从一直算到叶结点，另一个让所有结点都当root。
class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        # Edge Case:
        if not root:
            return 0

        # Process:
        def dfs(root, sum):
            count = 0
            if not root:
                return 0
            if root.val == sum:
                count += 1
            count += dfs(root.left, sum - root.val)
            count += dfs(root.right, sum - root.val)
            return count

        # recursion:
        return dfs(root, sum) + self.pathSum(root.left, sum) + self.pathSum(root.right, sum)


# 671. Second Minimum Node In a Binary Tree
"""
Given a non-empty special binary tree consisting of nodes with the non-negative value, where each node in this 
tree has exactly two or zero sub-node. If the node has two sub-nodes, then this node's value is the smaller value
 among its two sub-nodes. More formally, the property root.val = min(root.left.val, root.right.val) always holds.

Given such a binary tree, you need to output the second minimum value in the set made of all the nodes' value 
in the whole tree.

If no such second minimum value exists, output -1 instead.

Example 1:

Input: 
    2
   / \
  2   5
     / \
    5   7

Output: 5
Explanation: The smallest value is 2, the second smallest value is 5."""


class Solution(object):
    def findSecondMinimumValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.ans = float('inf')
        min1 = root.val

        def dfs(node):
            if node:
                if min1 < node.val < self.ans:
                    self.ans = node.val
                elif node.val == min1:
                    dfs(node.left)
                    dfs(node.right)

        dfs(root)
        if self.ans == float('inf'):
            return -1
        return self.ans

class Solution(object):
    def findSecondMinimumValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def find(root, val):
            if root == None:
                return val
            elif root.val != val:
                return root.val
            else:
                left_val = find(root.left, val)
                right_val = find(root.right, val)
                if min(left_val, right_val) == val:
                    return max(left_val, right_val)
                else:
                    return min(left_val, right_val)

        if root == None:
            return -1
        val = find(root, root.val)
        if val == root.val:
            return -1
        else:
            return val


class Solution(object):
    def findSecondMinimumValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        res = set()

        stack = [root]

        while stack != []:
            temp = stack.pop(0)
            res.add(temp.val)
            if temp.left != None:
                stack.append(temp.left)
            if temp.right != None:
                stack.append(temp.right)

        reslist = sorted(list(res))

        if len(reslist) < 2:
            return -1
        else:
            return reslist[1]


class Solution(object):
    def findSecondMinimumValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        res = [float('inf')]

        def traverse(node):
            if not node:
                return
            if root.val < node.val < res[0]:
                res[0] = node.val
            traverse(node.left)
            traverse(node.right)

        traverse(root)
        return -1 if res[0] == float('inf') else res[0]


# 572. Subtree of Another Tree
"""
Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with 
a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could 
also be considered as a subtree of itself.

Example 1:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
Given tree t:
   4 
  / \
 1   2
Return true, because t has the same structure and node values with a subtree of s."""
# faster
class Solution(object):
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        if t is None:return True
        elif s is None:return False
        def dfs(s):
            if s is None:
                return 'null'
            temp = str(s.val)+','+dfs(s.left)+',' + dfs(s.right)
            return ','+temp+','
        return dfs(s).find(dfs(t)) >= 0


class Solution(object):
    def isSubtree(self, s, t, findRoot=False):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        """
        solution:
        - Use recursion
        isSubTree(root_s, root_t) = {
                                    if root_s == root_t and isSubTree(root_s.left, root_t.left) and isSubTree(root_s.right, root_t.right): True
                                    elif findRoot and root_s != root_t: False
                                    isSubtree(s.left, t) or isSubtree(s.right, t)

        }
        """
        if s == None or t == None:
            return t == None and s == None

        # Handel s = [1, 1] t = [1]
        if s.val == t.val and self.isSubtree(s.left, t.left, True) and self.isSubtree(s.right, t.right, True):
            return True
        elif findRoot and s.val != t.val:
            return False
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)



def isMatch(self, s, t):
    if not(s and t):
        return s is t
    return (s.val == t.val and
            self.isMatch(s.left, t.left) and
            self.isMatch(s.right, t.right))

def isSubtree(self, s, t):
    if self.isMatch(s, t): return True
    if not s: return False
    return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

# method2

class Solution(object):
    def isSubtree(self, s, t):
        def isMatch(s, t):
            if (s is None and t is not None) or (s is not None and t is None):
                return False
            elif s is None and t is None:
                return True

            if s.val == t.val:
                if isMatch(s.left, t.left) and isMatch(s.right, t.right):
                    return True
                else:
                    return False

        if isMatch(s, t):
            return True
        if s is None:
            return False
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)


from collections import defaultdict

# 501. Find Mode in Binary Search Tree
"""
Given a binary search tree (BST) with duplicates, find all the mode(s) (the most frequently occurred element) in the given BST.

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than or equal to the node's key.
The right subtree of a node contains only nodes with keys greater than or equal to the node's key.
Both the left and right subtrees must also be binary search trees.
 

For example:
Given BST [1,null,2,2],

   1
    \
     2
    /
   2
 

return [2].
"""


class Solution(object):
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        self.dic = {}

        def nodeFrequency(node):
            if node:
                if node.val in self.dic:
                    self.dic[node.val] += 1
                else:
                    self.dic[node.val] = 1

                if node.left is None and node.right is None:
                    return
                else:
                    nodeFrequency(node.left)
                    nodeFrequency(node.right)

        nodeFrequency(root)
        modes = []
        maxVal = max(self.dic.values())
        for key in self.dic.keys():
            if self.dic[key] == maxVal:
                modes.append(key)
        return modes


class Solution(object):
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        self.ans = (None, 0)

        s = []
        if not root:
            return []
        if not root.right and not root.left:
            return [root.val]
        prev = None
        maxCount = 1
        curCount = 1
        ans = []

        while True:
            while root:
                s.append(root)
                root = root.left
            if not s:
                break
            root = s.pop()
            if root.val == prev:
                curCount += 1
            else:
                curCount = 1
            if curCount > maxCount:
                ans = [root.val]
                maxCount = curCount
            elif curCount == maxCount:
                ans.append(root.val)
            prev = root.val
            root = root.right
        return ans




from collections import defaultdict
class Solution(object):
    def helper(self, root, cache):
        if root == None:
            return
        cache[root.val] += 1
        self.helper(root.left, cache)
        self.helper(root.right, cache)
        return

    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root == None:
            return []
        cache = defaultdict(int)
        self.helper(root, cache)
        max_freq = max(cache.values())
        result = [k for k, v in cache.items() if v == max_freq]
        return result


# 654. Maximum Binary Tree
"""
Given an integer array with no duplicates. A maximum tree building on this array is defined as follow:

The root is the maximum number in the array.
The left subtree is the maximum tree constructed from left part subarray divided by the maximum number.
The right subtree is the maximum tree constructed from right part subarray divided by the maximum number.
Construct the maximum tree by the given array and output the root node of this tree.

Example 1:
Input: [3,2,1,6,0,5]
Output: return the tree root node representing the following tree:

      6
    /   \
   3     5
    \    / 
     2  0   
       \
        1
"""
"""
Algorithm:
We keep track of a stack, and make sure the numbers in stack is in decreasing order.

For each new num, we make it into a TreeNode first.
Then:

If stack is empty, we push the node into stack and continue
If new value is smaller than the node value on top of the stack, we append TreeNode as the right node of top of stack.
If new value is larger, we keep poping from the stack until the stack is empty OR top of stack node value is greater than the new value. During the pop, we keep track of the last node being poped.
After step 2, we either in the situation of 0, or 1, either way, we append last node as left node of the new node.
After traversing, the bottom of stack is the root node because the bottom is always the largest value we have seen so far (during the traversing of list).
"""
class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """

        if not nums:
            return None
        stk = []
        last = None
        for num in nums:
            while stk and stk[-1].val < num:
                last = stk.pop()
            node = TreeNode(num)
            if stk:
                stk[-1].right = node
            if last:
                node.left = last
            stk.append(node)
            last = None
        return stk[0]


class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """


        stack = []
        last_pop = None
        for i in nums:
            new_node = TreeNode(i)
            while stack and stack[-1].val < i:
                last_pop = stack.pop()
                new_node.left = last_pop

            if stack:
                stack[-1].right = new_node
            stack.append(new_node)
        return stack[0]

class Solution(object):
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if nums == []:
            return None
        m = max(nums)
        i = nums.index(m)
        root = TreeNode(m)
        root.left = self.constructMaximumBinaryTree(nums[:i])
        root.right = self.constructMaximumBinaryTree(nums[i+1:])
        return root