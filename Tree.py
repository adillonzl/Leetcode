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
        # Time: O(n)
        # Space: O(n)
        longest = [0]
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

#locked 250	Count Univalue Subtrees	postorder

#locked 366	Find Leaves of Binary Tree	postorder
#337	House Robber III	postorder + preorder
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