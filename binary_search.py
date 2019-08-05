"""
基础
69 square root
278	First Bad Version
875. Koko Eating Bananas
378. Kth Smallest Element in a Sorted Matrix
35	Search Insert Position
33	Search in Rotated Sorted Array
81	Search in Rotated Sorted Array II
153	Find Minimum in Rotated Sorted Array
154	Find Minimum in Rotated Sorted Array II
162	Find Peak Element
374	Guess Number Higher or Lower
34	Search for a Range
349	Intersection of Two Arrays
350	Intersection of Two Arrays II
315	Count of Smaller Numbers After Self
300	Longest Increasing Subsequence
354	Russian Doll Envelopes
"""
"""
二分一定要有序
要考虑元素是否有重复

binary search 模版
template:
[l,r） 左闭右开
def binary_search(l,r):
    while l < r:
        m = l + (r-l)//2
        if f(m): return m   # optional
        if g(m):  #这个函数根据题目来, 找到g(m) 为true
            r = m  # new range [l,m)
        else:
            l = m + 1 # new range [m+1,r)
    return l  # or not found

中间点  l + (r-l)//2
定义 左右和中间点，当左小于右是，比较目标和中间值，然后取一半，然后娜左或右的位子，取中间，再比较

我的弱点，找中间点的位子,偶数长度中间点在哪里, 注意找到mid是index还是值本身

首先，导致死循环的关键主要是L的变化方式，和hi的初始取值关系并不大。在你的模板里，把L=mid+1改成L=mid那就很可能会死循环。
因为使用 mid = (L+R)/2这种计算方式的话，当R-L=1时，mid是等于L的。而此时如果恰好执行了L=mid，那就意味着在这次iteration中，
L的值没有变化，即搜索范围没有变，于是就死循环了。

至于R的取值方式不同，更多地是反映出实现者的思路不同：如果取成nums.size()，则可能意味着你认为目标可能出现在[L, R)中；
hi取成nums.size()-1，意味着你认为目标一定会出现在[L, R]中。持前种思路的人，r = mid会更自然，而持后一种思路的人，
则更可能会写r=mid-1 (当然他写成r = mid也是一样可以的)。

一个有助于你快速判断是否会死循环的方法，是考虑R-L=1的情况。在这种情况下target可能有小于，等于A[L], 小于，等于，大于A[R]共5种情况。
快速验证一下这五种情况是否都能正常退出并返回正确值即可。

https://www.youtube.com/watch?v=v57lNF2mb_s
花花酱模版 左闭右开
[l,r)
1。 搞清楚是找target 还是找满足条件的最小边界
2。 左闭右开 [l,r) left = 0, right = n-1  （right = n-1是区间只有一个数的情况,eg [2], left= 0, right=0 mid=0）
3. while l < r or l <= r   (l <= r是区间只有一个数的情况）


如果右边求的是index，那一定要减1
while l <= r:               # 这里一定要有等于，考虑数组只有一个数的情况，否则进不了while 循环
eg，【5】，5 （704题）


最后一般return left
"""





# 69. Sqrt(x)
class Solution:
    """
    @param x: An integer
    @return: The sqrt of x
    """

    def sqrt(self, x):
        start, end = 1, x
        while start + 1 < end:
            mid = (start + end) / 2
            if mid * mid == x:
                return mid
            elif mid * mid < x:
                start = mid
            else:
                end = mid
        if end * end <= x:
            return end
        return start

    # method 2 Newton
    r = x
    while r * r > x:
        r = (r + x // r) // 2
    return r


# method 3 binary_search:

def sqrt2(self,x):
    l=0
    r = x+1       # 为什么这里+1
    while l < r:
        m = l + (r-1)//2
        if m*m >x:       #找到平方大于x的最小integer
            r = m       # 为什么这里不-1
        else:
            l = m + 1
    return l -1 


# 278	First Bad Version
class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        r = n-1
        l = 0
        while l<=r:
            mid = l + (r-l)/2
            if not isBadVersion(mid):
                l = mid+1
            else:
                r = mid-1
        return l


# 35	Search Insert Position
"""
Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You may assume no duplicates in the array.

Example 1:

Input: [1,3,5,6], 5
Output: 2
Example 2:

Input: [1,3,5,6], 2
Output: 1
Example 3:

Input: [1,3,5,6], 7
Output: 4
Example 4:

Input: [1,3,5,6], 0
Output: 0
"""
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        return len([x for x in nums if x<target])
#method2
def searchInsert(self, nums, target):
    return sorted(nums + [target]).index(target)


# method 3
def searchInsert(self, nums, target):
    for i in range(len(nums)):
        if target <= nums[i]:
            return i
    return len(nums)

# my binary search
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if nums is None:
            return 0

        l = 0
        r = len(nums)
        while l < r:
            mid = l + (r - l) // 2
            if target > nums[mid]:
                # res = mid
                l = mid + 1
            else:
                r = mid
        return r

    # 875. Koko Eating Bananas
"""find minimum K such that she can eat all the bananas within H hours
"""
#pseudo code:

def eat(piles,H):
    l = 1
    r = max(piles) + 1
    while l < r:
        m = l + (r-1)//2
        h = 0
        for p in piles:
            h =+ (p + m -1) / m
        if h < H:   # g(m): can finish
            r = m
        else:
            l = m + 1
    return l

"""
分析：

去找一个值满足某种条件，这种题见得太多了，显然是二分法，之后我整理一下所有的这种题目做一个合辑。
那么这里怎么选定初始的lo和hi呢？我们要明确我们找的是吃的速度，那么最低，起码得在吃吧，所以起码lo = 1，那hi呢？我们注意到note中第二点pile.length <= H，因为我们吃的速度就算再快，一次也只能吃一盘而已，所以无论怎样最少都得pile.length个小时才能吃完，所以hi = max(piles)
思路：

对于某个确定的k值，我们如何计算吃完所有pile需要的时间呢，对于一盘，时间应该是piles[i]/k 向上取整，然后求和判断是否大于H
若小于，则说明吃完还绰绰有余，还可以吃慢一点，从lo,mid中继续找
若大于，则说明吃得太慢了，则应该从mid,hi中继续找
"""
# O(NlogM) time，N = len(piles), M = max(piles)
class Solution(object):
    def minEatingSpeed(self, piles, H):
        """
        :type piles: List[int]
        :type H: int
        :rtype: int
        """
        lo,hi = 1,max(piles)
        def canEat(k):
            time = 0
            for i in range(len(piles)):
                time += int(math.ceil(piles[i]/float(k)))
                if time > H: return False
            return True
        while lo < hi:
            mid = (lo + hi) // 2
            if canEat(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

# method 3
def x():
    hi, lo = math.ceil(sum(piles) / (H - len(piles) + 1)), math.floor(sum(piles) / H)
    while hi - lo > 1:
        md = (hi + lo) // 2
        h = sum(math.ceil(pile / md) for pile in piles)
        if h > H:
                lo = md
        else:
                hi = md
    return hi


# 378. Kth Smallest Element in a Sorted Matrix
"""
Given a n x n matrix where each of the rows and columns are sorted in ascending order, 
find the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

Example:
matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,

return 13.
Note: You may assume k is always valid, 1 ≤ k ≤ n2.

matrix: List[List[int]]     注意这里不是个array没有shape。 [[5]] 的len 是1, len of above matrix is 3

"""


"""
pseudo code: 花花酱
def kthSamllest(A, K):
    l = A[0][0]
    r = A[-1][-1]
    while l < r:
        m = l + (r-1)//2
        total = 0 
        for row in A:
            total += upper_bound(row, m)
        if total >= k
            r = m
        else 
            l = m + 1
    return l
"""
"""
row = i// total columns
col = i % total rows
只有在第二行的第一个大于前一行的最后一个才成立
"""

class Solution(object):
    def kthSmallest(self, matrix, k):
        lo, hi = matrix[0][0], matrix[-1][-1]
        while lo<hi:
            mid = (lo+hi)//2
            if sum(bisect.bisect_right(row, mid) for row in matrix) < k: # 在每一行里做一个二分，找到小于k的个数 然后每行小于k的个数加起来
                lo = mid+1
            else:
                hi = mid
        return lo



import heapq
class Solution(object):
    def kthSmallest(self, matrix, k):
        return list(heapq.merge(*matrix))[k-1]

class Solution:
    def kthSmallest(self, matrix: 'List[List[int]]', k: 'int') -> 'int':
        left, right = matrix[0][0], matrix[-1][-1]
        while left < right:
            count = 0
            mid = (left+right)//2
            for i in range(len(matrix)):
                count += bisect.bisect_right(matrix[i], mid)
            if count >= k:
                right = mid
            else:
                left = mid + 1
        return left


def kthSmallest(self, matrix, k):
    heap = [(matrix[0][i], 0, i) for i in range(len(matrix[0]))]
    heapq.heapify(heap)
    for i in range(k - 1):
        v, r, c = heapq.heappop(heap)
        if r + 1 < len(matrix):
            heapq.heappush(heap, (matrix[r + 1][c], r + 1, c))
    return heapq.heappop(heap)[0]

def kthSmallest(self, matrix, k):
    """
    :type matrix: List[List[int]]
    :type k: int
    :rtype: int
    """
    if not matrix:
        return 0
    min_heap = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            heappush(min_heap, matrix[i][j])
    while k > 1:
        heappop(min_heap)
        k -= 1
    return heappop(min_heap)


class Solution(object):
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        return sorted(sum(matrix, []))[k-1]

"""
It's O(n) where n is the number of rows (and columns), not the number of elements. 
So it's very efficient. The algorithm is from the paper Selection in X + Y and 
matrices with sorted rows and columns, which I first saw mentioned by @elmirap 
(thanks).

The basic idea: Consider the submatrix you get by removing every second row and 
every second column. This has about a quarter of the elements of the original 
matrix. And the k-th element (k-th smallest I mean) of the original matrix is 
roughly the (k/4)-th element of the submatrix. So roughly get the (k/4)-th element 
of the submatrix and then use that to find the k-th element of the original matrix 
in O(n) time. It's recursive, going down to smaller and smaller submatrices until 
a trivial 2×2 matrix. For more details I suggest checking out the paper, the first 
half is easy to read and explains things well. Or @zhiqing_xiao's solution+
explanation.

Cool: It uses variants of saddleback search that you might know for example from 
the Search a 2D Matrix II problem. And it uses the median of medians algorithm for 
linear-time selection.

Optimization: If k is less than n, we only need to consider the top-left k×k matrix.
 Similar if k is almost n2. So it's even O(min(n, k, n^2-k)), I just didn't mention 
 that in the title because I wanted to keep it simple and because those few very 
 small or very large k are unlikely, most of the time k will be "medium" (and 
 average n2/2).

Implementation: I implemented the submatrix by using an index list through which 
the actual matrix data gets accessed. If [0, 1, 2, ..., n-1] is the index list of 
the original matrix, then [0, 2, 4, ...] is the index list of the submatrix and 
[0, 4, 8, ...] is the index list of the subsubmatrix and so on. This also covers
 the above optimization by starting with [0, 1, 2, ..., k-1] when applicable.

Application: I believe it can be used to easily solve the Find K Pairs with 
Smallest Sums problem in time O(k) instead of O(k log n), which I think is the 
best posted so far. I might try that later if nobody beats me to it (if you do, 
let me know :-). Update: I did that now."""


class Solution(object):
    def kthSmallest(self, matrix, k):

        # The median-of-medians selection function.
        def pick(a, k):
            if k == 1:
                return min(a)
            groups = (a[i:i + 5] for i in range(0, len(a), 5))
            medians = [sorted(group)[len(group) / 2] for group in groups]
            pivot = pick(medians, len(medians) / 2 + 1)
            smaller = [x for x in a if x < pivot]
            if k <= len(smaller):
                return pick(smaller, k)
            k -= len(smaller) + a.count(pivot)
            return pivot if k < 1 else pick([x for x in a if x > pivot], k)

        # Find the k1-th and k2th smallest entries in the submatrix.
        def biselect(index, k1, k2):

            # Provide the submatrix.
            n = len(index)

            def A(i, j):
                return matrix[index[i]][index[j]]

            # Base case.
            if n <= 2:
                nums = sorted(A(i, j) for i in range(n) for j in range(n))
                return nums[k1 - 1], nums[k2 - 1]

            # Solve the subproblem.
            index_ = index[::2] + index[n - 1 + n % 2:]
            k1_ = (k1 + 2 * n) / 4 + 1 if n % 2 else n + 1 + (k1 + 3) / 4
            k2_ = (k2 + 3) / 4
            a, b = biselect(index_, k1_, k2_)

            # Prepare ra_less, rb_more and L with saddleback search variants.
            ra_less = rb_more = 0
            L = []
            jb = n  # jb is the first where A(i, jb) is larger than b.
            ja = n  # ja is the first where A(i, ja) is larger than or equal to a.
            for i in range(n):
                while jb and A(i, jb - 1) > b:
                    jb -= 1
                while ja and A(i, ja - 1) >= a:
                    ja -= 1
                ra_less += ja
                rb_more += n - jb
                L.extend(A(i, j) for j in range(jb, ja))

            # Compute and return x and y.
            x = a if ra_less <= k1 - 1 else \
                b if k1 + rb_more - n * n <= 0 else \
                    pick(L, k1 + rb_more - n * n)
            y = a if ra_less <= k2 - 1 else \
                b if k2 + rb_more - n * n <= 0 else \
                    pick(L, k2 + rb_more - n * n)
            return x, y

        # Set up and run the search.
        n = len(matrix)
        start = max(k - n * n + n - 1, 0)
        k -= n * n - (n - start) ** 2
        return biselect(range(start, min(n, start + k)), k, k)[0]

# 35	Search Insert Position
"""
Given a sorted array and a target value, return the index if the target is found. 
If not, return the index where it would be if it were inserted in order.

You may assume no duplicates in the array.

Example 1:
Input: [1,3,5,6], 5
Output: 2

Example 2:
Input: [1,3,5,6], 2
Output: 1

Example 3:
Input: [1,3,5,6], 7
Output: 4

Example 4:
Input: [1,3,5,6], 0
Output: 0"""

class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        return len([x for x in nums if x<target])

class Solution(object):
def searchInsert(self, nums, key):
    if key > nums[len(nums) - 1]:
        return len(nums)

    if key < nums[0]:
        return 0

    l, r = 0, len(nums) - 1
    while l <= r:
        m = (l + r)/2
        if nums[m] > key:
            r = m - 1
            if r >= 0:
                if nums[r] < key:
                    return r + 1
            else:
                return 0

        elif nums[m] < key:
            l = m + 1
            if l < len(nums):
                if nums[l] > key:
                    return l
            else:
                return len(nums)
        else:
            return m

def searchInsert(self, nums, target): # works even if there are duplicates.
    l , r = 0, len(nums)-1
    while l <= r:
        mid=(l+r)/2
        if nums[mid] < target:
            l = mid+1
        else:
            if nums[mid]== target and nums[mid-1]!=target:
                return mid
            else:
                r = mid-1
    return l


# 33	Search in Rotated Sorted Array
"""Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1"""

class Solution:
    # @param {integer[]} numss
    # @param {integer} target
    # @return {integer}
    def search(self, nums, target):
        if not nums:
            return -1

        low, high = 0, len(nums) - 1

        while low <= high:
            mid = (low + high) / 2
            if target == nums[mid]:
                return mid

            if nums[low] <= nums[mid]:
                if nums[low] <= target <= nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if nums[mid] <= target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1

        return -1

def search(self, nums, target):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) / 2
        if (nums[0] > target) ^ (nums[0] > nums[mid]) ^ (target > nums[mid]):
            lo = mid + 1
        else:
            hi = mid
    return lo if target in nums[lo:lo+1] else -1


class Solution:
    def search(self, nums, target):
        self.__getitem__ = lambda i: \
            (nums[0] <= target) ^ (nums[0] > nums[i]) ^ (target > nums[i])
        i = bisect.bisect_left(self, True, 0, len(nums))
        return i if target in nums[i:i+1] else -1


# 81	Search in Rotated Sorted Array II
"""
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,0,1,2,2,5,6] might become [2,5,6,0,0,1,2]).

You are given a target value to search. If found in the array return true, otherwise return false.

Example 1:

Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true
Example 2:

Input: nums = [2,5,6,0,0,1,2], target = 3
Output: false
Follow up:

This is a follow up problem to Search in Rotated Sorted Array, where nums may contain duplicates.
Would this affect the run-time complexity? How and why?
"""
def search(self, nums, target):
    l, r = 0, len(nums)-1
    while l <= r:
        mid = l + (r-l)//2
        if nums[mid] == target:
            return True
        while l < mid and nums[l] == nums[mid]: # tricky part
            l += 1
        # the first half is ordered
        if nums[l] <= nums[mid]:
            # target is in the first half
            if nums[l] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        # the second half is ordered
        else:
            # target is in the second half
            if nums[mid] < target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
    return False


# 153	Find Minimum in Rotated Sorted Array
"""
https://www.youtube.com/watch?v=P4r7mF1Jd50
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

Find the minimum element.

You may assume no duplicate exists in the array.

Example 1:

Input: [3,4,5,1,2] 
Output: 1
Example 2:

Input: [4,5,6,7,0,1,2]
Output: 0
"""

class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        j = len(nums) - 1
        while i < j:
            m = i + (j - i) / 2
            if nums[m] > nums[j]:
                i = m + 1
            else:
                j = m
        return nums[i]

# recursive
class Solution:
    # @param {integer[]} nums
    # @return {integer}
    def findMin(self, nums):
        l = len(nums) - 1
        if l <= 1:
            return min(nums)
        mid = l / 2
        if nums[0] < nums[mid]:
            return self.findMin([nums[0]] + nums[mid+1:])
        else:
            return self.findMin(nums[1:mid+1])

class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1:
            return nums[0]
        n = len(nums)
        m = n / 2
        # first part from 0..m-1
        # second part from m..n-1
        if nums[0] <= nums[m-1] and nums[m] <= nums[n-1]:
            return min(nums[0], nums[m])
        elif nums[0] <= nums[m-1] and nums[m] >= nums[n-1]:
            return self.findMin(nums[m:n])
        else:
            return self.findMin(nums[:m])

# 154	Find Minimum in Rotated Sorted Array II hard
"""
153 with possible duplicates
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

Find the minimum element.

The array may contain duplicates.

Example 1:

Input: [1,3,5]
Output: 1
Example 2:

Input: [2,2,2,0,1]
Output: 0"""


def findMin(self, nums):
    l, r = 0, len(nums) - 1
    while l < r:
        m = (l + r) >> 1
        if nums[m] > nums[r]:
            l = m + 1
        elif nums[m] < nums[r]:
            r = m
        else:
            r = r - 1
    return nums[l]

# 162	Find Peak Element
"""
A peak element is an element that is greater than its neighbors.

Given an input array nums, where nums[i] ≠ nums[i+1], find a peak element and return its index.

The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

You may imagine that nums[-1] = nums[n] = -∞.

Example 1:

Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
Example 2:

Input: nums = [1,2,1,3,5,6,4]
Output: 1 or 5 
Explanation: Your function can return either index number 1 where the peak element is 2, 
             or index number 5 where the peak element is 6.
Note:

Your solution should be in logarithmic complexity."""


# O(n) time
def findPeakElement1(self, nums):
    i = 0
    while i <= len(nums) - 1:
        while i < len(nums) - 1 and nums[i] < nums[i + 1]:
            i += 1
        return i

    # O(lgn) time


def findPeakElement2(self, nums):
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] > nums[mid + 1]:
            r = mid
        else:
            l = mid + 1
    return l


# Recursively
def findPeakElement(self, nums):
    return self.helper(nums, 0, len(nums) - 1)


def helper(self, nums, l, r):
    if l == r:
        return l
    mid = l + (r - l) // 2
    if nums[mid] > nums[mid + 1]:
        return self.helper(nums, l, mid)
    else:
        return self.helper(nums, mid + 1, r)


# O(n) time
def findPeakElement(self, nums):
    if not nums:
        return 0
    nums.insert(0, -(sys.maxint + 1))
    nums.append(-(sys.maxint + 1))
    for i in xrange(1, len(nums) - 1):
        if nums[i - 1] < nums[i] > nums[i + 1]:
            return i - 1


# O(lgn) time
def findPeakElement(self, nums):
    if not nums:
        return 0
    l, r = 0, len(nums) - 1
    while l <= r:
        if l == r:
            return l
        mid = l + (r - l) // 2
        # due to "mid" is always the left one if the length of the list is even,
        # so "mid+1" is always valid.
        if (mid - 1 < 0 or nums[mid - 1] < nums[mid]) and nums[mid] > nums[mid + 1]:
            return mid
        elif nums[mid] > nums[mid + 1]:
            r = mid
        else:
            l = mid + 1



"""类型：考虑边界 
Time Complexity (logN )
Time Spent on this question: 50 mins | 重刷
这题套惯用的模板要考虑Edge Case，因为判定条件是:

上坡：nums[mid-1] < nums[mid]
下坡： nums[mid] > nums[mid+1]

然后这里mid + 1 和 mid - 1在数组大小为 1 或者 2的时候，很容易就越界了，所以在大小范围增加最小值和最大值。得到：
if (mid == 0 or nums[mid-1] < nums[mid]) and (mid == len(nums) - 1 or nums[mid] > nums[mid+1]):

"""
class Solution(object):
    def findPeakElement(self, nums):
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = l + (r-l) // 2
            if (mid == 0 or nums[mid-1] < nums[mid]) and (mid == len(nums) - 1 or nums[mid] > nums[mid+1]):
                return mid;
            if nums[mid] < nums[mid+1]:
                l = mid + 1
            else:
                r = mid - 1
#这道题目也可以套另外一个模板，可以不用考虑越界的问题：
class Solution(object):
    def findPeakElement(self, nums):
        l, r = 0, len(nums) - 1
        while l < r:
            mid = l + (r-l) // 2
            if nums[mid] < nums[mid+1]:
                l = mid + 1
            else:
                r = mid
        return l

#374	Guess Number Higher or Lower
"""We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I'll tell you whether the number is higher or lower.

You call a pre-defined API guess(int num) which returns 3 possible results (-1, 1, or 0):

-1 : My number is lower
 1 : My number is higher
 0 : Congrats! You got it!
Example :

Input: n = 10, pick = 6
Output: 6
"""
class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        low = 1
        high = n
        while low <= high:
            mid = (low + high)//2
            res =  guess(mid)
            if res == 0 :
                return mid
            elif res == -1:
                high = mid - 1
            else:
                low = mid + 1


class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        l, r = 1, n
        while l + 1 < r:
            m = l + (r - l) / 2
            res = guess(m)
            if res < 0:
                r = m
            elif res > 0:
                l = m
            else:
                return m

        if guess(l) == 0:
            return l
        if guess(r) == 0:
            return r
        return None

#34	Find First and Last Position of Element in Sorted Array
"""
Given an array of integers nums sorted in ascending order, find the starting and ending position
 of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].

Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Example 2:

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
"""
# solution 1
def searchRange(self, nums, target):
    def search(lo, hi):
        if nums[lo] == target == nums[hi]:
            return [lo, hi]
        if nums[lo] <= target <= nums[hi]:
            mid = (lo + hi) / 2
            l, r = search(lo, mid), search(mid+1, hi)
            return max(l, r) if -1 in l+r else [l[0], r[1]]
        return [-1, -1]
    return search(0, len(nums)-1)
"""The O(log n) time isn't quite obvious, so I'll explain it below. Or you can take the challenge
 and prove it yourself :-)
The search helper function returns an index range just like the requested searchRange function, 
but only searches in nums[lo..hi]. It first compares the end points and immediately returns [
lo, hi] if that whole part of nums is full of target, and immediately returns [-1, -1] if target
 is outside the range. The interesting case is when target can be in the range but doesn't fill 
 it completely.

In that case, we split the range in left and right half, solve them recursively, and combine 
their results appropriately. Why doesn't this explode exponentially? Well, let's call the numbers
 in the left half A, ..., B and the numbers in the right half C, ..., D. Now if one of them 
 immediately return their [lo, hi] or [-1, -1], then this doesn't explode. And if neither 
 immediately returns, that means we have A <= target <= B and C <= target <= D. And since nums 
 
 is sorted, we actually have target <= B <= C <= target, so B = C = target. The left half thus 
 ends with target and the right half starts with it. I highlight that because it's important. 
 Now consider what happens further. The left half gets halved again. Call the middle elements a 
 and b, so the left half is A, ..., a, b, ..., B. Then a <= target and:

If a < target, then the call analyzing A, ..., a immediately returns [-1, -1] and we only look 
further into b, ..., B which is again a part that ends with target.
If a == target, then a = b = ... = B = target and thus the call analyzing b, ..., B immediately 
returns its [lo, hi] and we only look further into A, ..., a which is again a part that ends with target.
Same for the right half C, ..., D. So in the beginning of the search, as long as target is only 
in at most one of the two halves (so the other immediately stops), we have a single path. And if 
we ever come across the case where target is in both halves, then we split into two paths, but 
then each of those remains a single path. And both paths are only O(log n) long, so we have 
overall runtime O(log n).

"""
# Solution 2 : Two binary searches : 56 ms
def searchRange(self, nums, target):
    def search(n):
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) / 2
            if nums[mid] >= n:
                hi = mid
            else:
                lo = mid + 1
        return lo
    lo = search(target)
    return [lo, search(target+1)-1] if target in nums[lo:lo+1] else [-1, -1]
"""
Here, my helper function is a simple binary search, telling me the first index where I could 
insert a number n into nums to keep it sorted. Thus, if nums contains target, I can find the 
first occurrence with search(target). I do that, and if target isn't actually there, then I 
return [-1, -1]. Otherwise, I ask search(target+1), which tells me the first index where I could
 insert target+1, which of course is one index behind the last index containing target, so all 
 I have left to do is subtract 1."""

#Solution 3 : Two binary searches, using the library
def searchRange(self, nums, target):
    lo = bisect.bisect_left(nums, target)
    return [lo, bisect.bisect(nums, target)-1] if target in nums[lo:lo+1] else [-1, -1]


# variant: search sorted array for 1st occurence of K
"""https://www.youtube.com/watch?time_continue=471&v=gOkNq8Co6B8 
Space : iterative O(1), recursive O(log(n)) ??
"""


#349	Intersection of Two Arrays
"""Given two arrays, write a function to compute their intersection.

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2]
Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [9,4]
Note:

Each element in the result must be unique.
The result can be in any order."""
#Solution 1:use set operation in python, one-line solution.
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        return list(set(nums1) & set(nums2))

# brute-force searching, search each element of the first list in the second list.
# (to be more efficient,
# you can sort the second list and use binary search to accelerate)
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        res = []
        for i in nums1:
            if i not in res and i in nums2:
                res.append(i)

        return res

## my binary (didn't pass)
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        # return set(set(nums1) &set(nums2))
        result = []
        nums1 = nums1.sort()
        nums2 = nums2.sort()
        mid = len(nums2) // 2
        for i in nums1:

            if i < nums2[mid]:
                nums2_half = nums2[:mid]
            else:
                nums2_half = nums2[mid:]
            if i not in result and i in nums2_half:
                result.append(i)

        return result



# Solution 3: use dict/hashmap to record all nums appeared in the first list, and then
# check if there are nums in the second list have appeared in the map.


class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        res = []
        map = {}
        for i in nums1:
            map[i] = map[i] + 1 if i in map else 1
        for j in nums2:
            if j in map and map[j] > 0:
                res.append(j)
                map[j] = 0

        return res

# Solution 4: sort the two list, and use two pointer to search in the lists to find common elements.

class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        res = []
        nums1.sort()
        nums2.sort()
        i = j = 0
        while (i < len(nums1) and j < len(nums2)):
            if nums1[i] > nums2[j]:
                j += 1
            elif nums1[i] < nums2[j]:
                i += 1
            else:
                if not (len(res) and nums1[i] == res[len(res) - 1]):
                    res.append(nums1[i])
                i += 1
                j += 1

        return res


#350	Intersection of Two Arrays II
"""Given two arrays, write a function to compute their intersection.

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]
Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [4,9]
Note:

Each element in the result should appear as many times as it shows in both arrays.
The result can be in any order.
Follow up:

What if the given array is already sorted? How would you optimize your algorithm?
What if nums1's size is small compared to nums2's size? Which algorithm is better?
What if elements of nums2 are stored on disk, and the memory is limited such that 
you cannot load all elements into the memory at once?
"""
def intersect(self, nums1, nums2):
    a, b = map(collections.Counter, (nums1, nums2))
    return list((a & b).elements())


def intersect(self, nums1, nums2):
    C = collections.Counter
    return list((C(nums1) & C(nums2)).elements())


def intersect(self, nums1, nums2):
    return list((collections.Counter(nums1) & collections.Counter(nums2)).elements())

class Solution(object):
    def intersect(self, nums1, nums2):

        nums1, nums2 = sorted(nums1), sorted(nums2)
        pt1 = pt2 = 0
        res = []

        while True:
            try:
                if nums1[pt1] > nums2[pt2]:
                    pt2 += 1
                elif nums1[pt1] < nums2[pt2]:
                    pt1 += 1
                else:
                    res.append(nums1[pt1])
                    pt1 += 1
                    pt2 += 1
            except IndexError:
                break

        return res

"""
744. Find Smallest Letter Greater Than Target
Easy

Share
Given a list of sorted characters letters containing only lowercase letters, and given a target letter target, find the smallest element in the list that is larger than the given target.

Letters also wrap around. For example, if the target is target = 'z' and letters = ['a', 'b'], the answer is 'a'.
"""
class Solution(object):
    def nextGreatestLetter(self, letters, target):
        seen = set(letters)
        for i in xrange(1, 26):
            cand = chr((ord(target) - ord('a') + i) % 26 + ord('a'))
            if cand in seen:
                return cand



class Solution(object):
    def nextGreatestLetter(self, letters, target):
        for c in letters:
            if c > target:
                return c
        return letters[0]

# BS， need sorted
class Solution(object):
    def nextGreatestLetter(self, letters, target):
        index = bisect.bisect(letters, target)
        return letters[index % len(letters)]


class Solution(object):
    def nextGreatestLetter(self, letters, target):
        """
        :type letters: List[str]
        :type target: str
        :rtype: str
        """
        n = len(letters)
        if n == 0:
            return None

        low = 0
        high = n - 1
        # If it can not be found, must be the first element (wrap around)
        result = 0

        while low <= high:
            mid = low + (high - low) // 2
            if letters[mid] > target:
                result = mid
                high = mid - 1
            else:
                low = mid + 1

        return letters[result]





def binary_search(l,r):
    while l<r:
        m = l + (r-l)//2
        if f(m): return m  # 判断m是不是解， 并不是所有题都需要。 optional
        if A[mid]>target:         # 判断左右边，注意是大于 还是大于等于
            r = m # new range [l,r)
        else:
            l = m + 1  # new range [m+1, r)

    return l


# 367
class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        l = 0
        r = num

        while l <= r:    # 我的难点why = here?
            mid = l + (r - l) // 2
            if mid ** 2 == num:
                return True
            elif mid * mid > num:
                r = mid - 1
            else:
                l = mid + 1

        return False



# 374 passed
class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        low = 0
        hi = n
        while low < hi:
            mid = low + (hi-low)//2
            if guess(mid)==1:
                low = mid + 1
            else:
                hi = mid    # 我的难点：为什么不考虑==0 直接return low
        return low



# 441
def arrangeCoins(self, n):
    """
    :type n: int
    :rtype: int
    """
    left = 0
    right = n

    while left <= right:

        mid = left + (right - left) // 2

        # sum of n natural numbers equation
        total = mid * (mid + 1) / 2

        if n >= total:
            left = mid + 1
            ans = mid
        else:
            right = mid - 1

    return ans


# 475 heater
class Solution:
    def findRadius(self, houses: 'List[int]', heaters: 'List[int]') -> 'int':
        n = len(heaters)
        heaters = sorted(heaters)

        def binary_search(start, end, target):

            # Why <= instead of < ?
            # Examine test case and see for yourself:
            # [58], [40, 65, 92, 42, 87, 3, 27, 29, 40, 12]
            while start <= end:
                mid = (start + end) // 2
                if heaters[mid] == target:
                    return 0
                elif heaters[mid] > target:
                    end = mid - 1
                elif heaters[mid] < target:
                    start = mid + 1
            after = mid + 1 if mid < n - 1 else mid
            before = mid - 1 if mid > 0 else mid
            return min(abs(heaters[after] - target), abs(heaters[before] - target), abs(heaters[mid] - target))

        return max(binary_search(0, n - 1, value) for value in houses)


class Solution:
    def findRadius(self, houses, heaters):
        heaters.sort()
        r = 0
        for h in houses:
            ind = bisect.bisect_left(heaters, h)
            if ind == len(heaters):
                r = max(r, h - heaters[-1])
            elif ind == 0:
                r = max(r, heaters[0] - h)
            else:
                r = max(r, min(heaters[ind] - h, h - heaters[ind - 1]))
        return r

def findRadius(self, houses, heaters):
    heaters = sorted(heaters) + [float('inf')]
    i = r = 0
    for x in sorted(houses):
        while x >= sum(heaters[i:i+2]) / 2.:
            i += 1
        r = max(r, abs(heaters[i] - x))
    return r


#704
class Solution:
    def search(self, nums, target):
        l, r = 0, len(nums) - 1     # 如果右边求的是index，那一定要减1
        while l <= r:               # 这里一定要有等于，考虑数组只有一个数的情况，否则进不了while 循环
            mid = (l + r) // 2
            if nums[mid] < target:
                l = mid + 1
            elif nums[mid] > target:
                r = mid - 1
            else:
                return mid
        return -1

# 287 Find the Duplicate Number
class Solution(object):
    def findDuplicate(self, nums):
        low = 0
        high = len(nums) - 1
        mid = (high + low) / 2
        while high - low > 1:
            count = 0
            for k in nums:
                if mid < k <= high:
                    count += 1
            if count > high - mid:
                low = mid
            else:
                high = mid
            mid = (high + low) / 2
        return high

# my solution
def findDuplicate(self, nums):
    left = 0
    right = len(nums) - 1
    nums.sort()
    while right - left > 1:   # 这里如果是left < right, 在最后两个[2,2]会死循环
        # creat a g(m)
        mid = left + (right - left) // 2
        print
        left, right
        if nums[mid] - nums[left] < (mid - left):
            right = mid
        elif nums[mid] - nums[left] >= (mid - left):
            left = mid

    return nums[left]


# 454 4Sum II
def fourSumCount(self, A, B, C, D):
    AB = collections.Counter(a+b for a in A for b in B)
    return sum(AB[-c-d] for c in C for d in D)


def fourSumCount(self, A, B, C, D):
    hashtable = {}
        for a in A:
            for b in B :
                if a + b in hashtable :
                    hashtable[a+b] += 1
                else :
                    hashtable[a+b] = 1
        count = 0
        for c in C :
            for d in D :
                if -c - d in hashtable :
                    count += hashtable[-c-d]
        return count

def fourSumCount(self, A, B, C, D):
    total = {}
        for a in A:
            for b in B:
                if a+b in total:
                    total[a+b] += 1
                else:
                    total[a+b] = 1
        ans = 0
        for c in C:
            for d in D:
                if -c-d in total:
                    ans += total[-c-d]
        return ans

def fourSumCount(self, A, B, C, D):
    ab = {}
    for i in A:
        for j in B:
            ab[i + j] = ab.get(i + j, 0) + 1

    ans = 0
    for i in C:
        for j in D:
            ans += ab.get(-i - j, 0)
    return ans


#392. Is Subsequence
"""
s = "abc", t = "zyahbgdc"
"""
import collections
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        d = collections.defaultdict(list)  # still key, value dictionary, but value is a list, so element append to the list based on key
        for i in xrange(0, len(t)):
            d[t[i]].append(i)
        start = 0
        for c in s:
            idx = bisect.bisect_left(d[c], start)   # bisect_left(list, target) return left most possible index to insert
            # start to dictionary 中每个list的index
            # defaultdict( < type
            # 'list' >, {u'a': [2, 6], u'c': [8], u'b': [4], u'd': [7], u'g': [5], u'h': [3], u'y': [1], u'z': [0]})
            if len(d[c]) == 0 or idx >= len(d[c]):
                return False
            start = d[c][idx] + 1
        return True


class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        idx = collections.defaultdict(list)
        for i, c in enumerate(t):
            idx[c].append(i)
        prev = 0
        for i, c in enumerate(s):
            j = bisect.bisect_left(idx[c], prev)
            if j == len(idx[c]): return False
            prev = idx[c][j] + 1
        return True
