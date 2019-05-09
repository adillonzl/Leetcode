"""
一维， 2 sets of sub_problems
70 Climbing Stairs
926
845
62	Unique Paths
63	Unique Paths II
120	Triangle	很少考
279	Perfect Squares
139	Word Break
375	Guess Number Higher or Lower II
312	Burst Balloons
322	Coin Change
91
509

1D with multiple states
801
926
790
818


二维
256	Paint House
265	Paint House II
64	Minimum Path Sum
72	Edit Distance
97	Interleaving String
174	Dungeon Game
221	Maximal Square
85	Maximal Rectangle
363	Max Sum of Rectangle No Larger Than K
化简
198 House Robber
213	House Robber II
276	Paint Fence
91	Decode Ways
10	Regular Expression Matching
44	Wildcard Matching
"""
#70. Climbing Stairs
"""https://www.youtube.com/watch?v=3mY5W0yojtA

You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Note: Given n will be a positive integer.

Example 1:

Input: 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
"""

class Solution:
    """
    @param n: An integer
    @return: An integer
    """

    def climbStairs(self, n):
        # write your code here
        if n == 0:
            return 1
        if n <= 2:
            return n
        result = [1, 2]
        for i in range(n - 2):
            result.append(result[-2] + result[-1])
        return result[-1]

# 变形 爬楼梯II， 每次可以1，2，3步
class Solution:
    """
    @param {int} n a integer
    @return {int} a integer
    """

    def climbStairs2(self, n):
        # write your code here
        if n <= 1:
            return 1

        if n == 2:
            return 2

        a, b, c = 1, 1, 2
        for i in range(3, n + 1):
            a, b, c = b, c, a + b + c

        return c


# 746. Min Cost Climbing Stairs
# On a staircase, the i-th step has some non-negative cost cost[i] assigned (0 indexed).
# Once you pay the cost, you can either climb one or two steps. You need to find minimum
# cost to reach the top of the floor, and you can either start from the step with index 0,
# or the step with index 1.
def minCostClimbingStairs(self, cost):
    min_cost0, min_cost1 = cost[0], cost[1]
    for c in cost[2:]:
        min_cost0, min_cost1 = min_cost1, min(min_cost0, min_cost1) + c
    return min(min_cost0, min_cost1)


class Solution:
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        for i in range(2, len(cost)):
            cost[i] = min(cost[i] + cost[i - 1], cost[i] + cost[i - 2]);
        return min(cost[-1], cost[-2]);


# 926. Flip String to Monotone Increasing
"""A string of '0's and '1's is monotone increasing if it consists of some number of '0's (possibly 0), followed by some number of '1's (also possibly 0.)

We are given a string S of '0's and '1's, and we may flip any '0' to a '1' or a '1' to a '0'.

Return the minimum number of flips to make S monotone increasing.

 

Example 1:

Input: "00110"
Output: 1
Explanation: We flip the last digit to get 00111.
Example 2:

Input: "010110"
Output: 2
Explanation: We flip to get 011111, or alternatively 000111.
Example 3:

Input: "00011000"
Output: 2
Explanation: We flip to get 00000000.
 

Note:

1 <= S.length <= 20000
S only consists of '0' and '1' characters.
"""
"""思路分析：
假设经过最优解的翻转使其变成了 s = '0'*i + '1'*j
其实我们要决定的是在原字符串中选择哪一个位置的’1’，使其作为最优解中的第一个’1’！

例如对于示例2中的010110，假设我们选择index=1处的’1’作为开头的’1’，那么我们需要将后面
所有的’0’全部翻转成’1’，翻转次数取决于后面’0’的个数。

又假设我们选取index=3处的’1’作为开头的’1’，那么我们需要将前面所有的'1'翻转成0，将后面
所有的'0'翻转成'1'，翻转次数取决于前面’1’的个数和后面’0’的个数。

OK，那我们只要一直记录着当前位置的前面有多少个1，后面有多少个0即可！
注意：可能存在最后全为0的情况（如示例3），那么我们是选不到某个’1’作为开头的，所以我们先
将全部翻转成’0’所花费的次数作为初始默认次数，然后和我们每次计算的结果比较即可。
"""

class Solution(object):
    def minFlipsMonoIncr(self, s):
        n = len(s)
        cnt0 = s.count('0')
        cnt1 = 0
        res = n - cnt0
        for i in range(n):
            if s[i] == '0':
                cnt0 -= 1
            elif s[i] == '1':
                res = min(res, cnt1+cnt0)
                cnt1 += 1
        return res

# 845. Longest Mountain in Array
"""Let's call any (contiguous) subarray B (of A) a mountain if the following properties
 hold:

B.length >= 3
There exists some 0 < i < B.length - 1 such that B[0] < B[1] < ... B[i-1] < B[i] > B[i+1] > ... > B[B.length - 1]
(Note that B could be any subarray of A, including the entire array A.)

Given an array A of integers, return the length of the longest mountain. 

Return 0 if there is no mountain.

Example 1:

Input: [2,1,4,7,3,2,5]
Output: 5
Explanation: The largest mountain is [1,4,7,3,2] which has length 5.
Example 2:

Input: [2,2,2]
Output: 0
Explanation: There is no mountain.
Note:

0 <= A.length <= 10000
0 <= A[i] <= 10000
Follow up:

Can you solve it using only one pass?
Can you solve it in O(1) space?

"""
"""Intuition:
We have already many 2-pass or 3-pass problems, like 821. Shortest Distance to a Character.
They have almost the same idea.
One forward pass and one backward pass.
Maybe another pass to get the final result, or you can merge it in one previous pass.

Explanation:
In this problem, we take one forward pass to count up hill length (to every point).
We take another backward pass to count down hill length (from every point).
Finally a pass to find max(up[i] + down[i] + 1) where up[i] and down[i] should be positives.

Time Complexity:
O(N)
"""

def longestMountain(self, A):
    res = up = down = 0
    for i in range(1, len(A)):
        if down and A[i - 1] < A[i] or A[i - 1] == A[i]: up = down = 0
        up += A[i - 1] < A[i]
        down += A[i - 1] > A[i]
        if up and down: res = max(res, up + down + 1)
    return res

class Solution:
    def longestMountain(self, A, res = 0):
        for i in range(1, len(A) - 1):
            if A[i + 1] < A[i] > A[i - 1]:
                l = r = i
                while l and A[l] > A[l - 1]: l -= 1
                while r + 1 < len(A) and A[r] > A[r + 1]: r += 1
                if r - l + 1 > res: res = r - l + 1
        return res

# 821. Shortest Distance to a Character
"""Given a string S and a character C, return an array of integers representing the 
shortest distance from the character C in the string.

Example 1:

Input: S = "loveleetcode", C = 'e'
Output: [3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]
 

Note:

S string length is in [1, 10000].
C is a single character, and guaranteed to be in string S.
All letters in S and C are lowercase."""

class Solution:
    def shortestToChar(self, S, C):
        """
        :type S: str
        :type C: str
        :rtype: List[int]
        """
        c = []
        for i, v in enumerate(S):
            if v == C:
                c.append(i)

        r = []
        for i in range(len(S)):
            r.append(min([abs(t - i)for t in c]))
        return r


# Idea is to move from left to right, right to left and update distance related
# to seen character
class Solution:
    def shortestToChar(self, s, c):
        res = [float("inf")] * len(s)
        dx = 1
        cur = i = 0
        while i > -1:
            if cur:
                res[i] = min(res[i], cur)
                cur += 1
            if s[i] == c:
                res[i] = 0
                cur = 1
            i += dx
            if i == len(s):
                dx = -1
                cur = 0
                i -= 1
        return res

# 62. Unique Paths
"""A robot is located at the top-left corner of a m x n grid (marked 'Start' in 
the diagram below).

The robot can only move either down or right at any point in time. The robot is 
trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram 
below).

How many possible unique paths are there?
Note: m and n will be at most 100.

Example 1:

Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right
Example 2:

Input: m = 7, n = 3
Output: 28

"""
class Solution:
    # @return an integer
    def uniquePaths(self, m, n):
        aux = [[1 for x in range(n)] for x in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                aux[i][j] = aux[i][j-1]+aux[i-1][j]
        return aux[-1][-1]


# math C(m+n-2,n-1)
def uniquePaths1(self, m, n):
    if not m or not n:
        return 0
    return math.factorial(m + n - 2) / (math.factorial(n - 1) * math.factorial(m - 1))


# O(m*n) space
def uniquePaths2(self, m, n):
    if not m or not n:
        return 0
    dp = [[1 for _ in xrange(n)] for _ in xrange(m)]
    for i in xrange(1, m):
        for j in xrange(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[-1][-1]


# O(n) space
def uniquePaths(self, m, n):
    if not m or not n:
        return 0
    cur = [1] * n
    for i in xrange(1, m):
        for j in xrange(1, n):
            cur[j] += cur[j - 1]
    return cur[-1]


# 63	Unique Paths II
"""
Example 1:

Input:
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
Output: 2
Explanation:
There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right

"""


# O(m*n) space
def uniquePathsWithObstacles1(self, obstacleGrid):
    if not obstacleGrid:
        return
    r, c = len(obstacleGrid), len(obstacleGrid[0])
    dp = [[0 for _ in xrange(c)] for _ in xrange(r)]
    dp[0][0] = 1 - obstacleGrid[0][0]
    for i in xrange(1, r):
        dp[i][0] = dp[i - 1][0] * (1 - obstacleGrid[i][0])
    for i in xrange(1, c):
        dp[0][i] = dp[0][i - 1] * (1 - obstacleGrid[0][i])
    for i in xrange(1, r):
        for j in xrange(1, c):
            dp[i][j] = (dp[i][j - 1] + dp[i - 1][j]) * (1 - obstacleGrid[i][j])
    return dp[-1][-1]


# O(n) space
def uniquePathsWithObstacles2(self, obstacleGrid):
    if not obstacleGrid:
        return
    r, c = len(obstacleGrid), len(obstacleGrid[0])
    cur = [0] * c
    cur[0] = 1 - obstacleGrid[0][0]
    for i in xrange(1, c):
        cur[i] = cur[i - 1] * (1 - obstacleGrid[0][i])
    for i in xrange(1, r):
        cur[0] *= (1 - obstacleGrid[i][0])
        for j in xrange(1, c):
            cur[j] = (cur[j - 1] + cur[j]) * (1 - obstacleGrid[i][j])
    return cur[-1]


# in place
def uniquePathsWithObstacles(self, obstacleGrid):
    if not obstacleGrid:
        return
    r, c = len(obstacleGrid), len(obstacleGrid[0])
    obstacleGrid[0][0] = 1 - obstacleGrid[0][0]
    for i in xrange(1, r):
        obstacleGrid[i][0] = obstacleGrid[i - 1][0] * (1 - obstacleGrid[i][0])
    for i in xrange(1, c):
        obstacleGrid[0][i] = obstacleGrid[0][i - 1] * (1 - obstacleGrid[0][i])
    for i in xrange(1, r):
        for j in xrange(1, c):
            obstacleGrid[i][j] = (obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1]) * (1 - obstacleGrid[i][j])
    return obstacleGrid[-1][-1]

