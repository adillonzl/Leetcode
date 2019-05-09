第一遍
看答案，学数据结构，学最优解
第二遍
背
第三遍
自己写


# 28. Implement strStr()
def strStr(self, haystack, needle):
    """
    :type haystack: str
    :type needle: str
    :rtype: int
    """
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


# 459. Repeated Substring Pattern


# 771. Jewels and Stones
class a(object):
    def numJewelsInStones(self, J, S):
        return sum(s in J for s in S)


# 242. Valid Anagram
# 写出一个函数 anagram(s, t) 判断两个字符串是否可以通过改变字母的顺序变成一样的字符串
class Solution:
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        return sorted(s) == sorted(t)


# 438. Find All Anagrams in a String
# return all possible index into a list
# Given a string s and a non-empty string p, find all the start indices of p's anagrams in s
from collections import Counter


def findAnagrams(self, s, p):
    """
    :type s: str
    :type p: str
    :rtype: List[int]
    """
    res = []
    pCounter = Counter(p)
    sCounter = Counter(s[:len(p) - 1])
    for i in range(len(p) - 1, len(s)):
        sCounter[s[i]] += 1  # include a new char in the window
        if sCounter == pCounter:  # This step is O(1), since there are at most 26 English letters
            res.append(i - len(p) + 1)  # append the starting index
        sCounter[s[i - len(p) + 1]] -= 1  # decrease the count of oldest char in the window
        if sCounter[s[i - len(p) + 1]] == 0:
            del sCounter[s[i - len(p) + 1]]  # remove the count if it is 0
    return res


class Solution:
    # @param {string} s a string
    # @param {string} p a non-empty string
    # @return {int[]} a list of index
    def findAnagrams(self, s, p):
        # Write your code here
        ans = []
        sum = [0 for x in range(0, 30)]
        plength = len(p)
        slength = len(s)
        for i in range(plength):
            sum[ord(p[i]) - ord('a')] += 1
        start = 0
        end = 0
        matched = 0
        while end < slength:
            if sum[ord(s[end]) - ord('a')] >= 1:
                matched += 1
            sum[ord(s[end]) - ord('a')] -= 1
            end += 1
            if matched == plength:
                ans.append(start)
            if end - start == plength:
                if sum[ord(s[start]) - ord('a')] >= 0:
                    matched -= 1
                sum[ord(s[start]) - ord('a')] += 1
                start += 1
        return ans


# 760. Find Anagram Mappings
class Solution(object):
    def anagramMappings(self, A, B):
        D = {x: i for i, x in enumerate(B)}
        return [D[x] for x in A]


# 804. Unique Morse Code Words
class Solution:
    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        morse = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--",
                 "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
        s = set()

        s = {"".join(morse[ord(c) - ord('a')] for c in word)
             for word in words}

        return len(s)


# 3. Longest Substring Without Repeating Characters （medium)
# Given "pwwkew", the answer is "wke", with the length of 3. Note that
# the answer must be a substring, "pwke" is a subsequence and not a substring.

# 不一定从头开始的，可能是中间一节

def lengthOfLongestSubstring(self, s):
    longest = []
    max_length = 0

    for c in s:
        if c in longest:
            max_length = max(max_length, len(longest))
            longest = longest[longest.index(c) + 1:]
        longest.append(c)

    max_length = max(max_length, len(longest))
    return max_length


def lengthOfLongestSubstring(self, s):
    """
    """
    # runtime: 95ms
    dic = {}
    res, last_match = 0, -1
    for i, c in enumerate(s):
        if c in dic and last_match < dic[c]:
            last_match = dic[c]
        res = max(res, i - last_match)
        dic[c] = i
    return res


# 387. First Unique Character in a String
# Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.
# s = "leetcode" return 0.
# s = "loveleetcode", return 2.
def firstUniqChar(self, s):
    """
    :type s: str
    :rtype: int
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    index = [s.index(l) for l in letters if s.count(l) == 1]  # in letters not in s
    return min(index) if len(index) > 0 else -1


class Solution(object):
    def firstUniqChar(self, s):
        return min([s.find(c) for c in string.ascii_lowercase if s.count(c) == 1] or [-1])


# 859. Buddy Strings
# Given two strings A and B of lowercase letters, return true if and
# only if we can swap two letters in A so that the result equals B
# https://leetcode.com/problems/buddy-strings/description/
def buddyStrings(self, A, B):
    if len(A) != len(B): return False
    if A == B and len(set(A)) < len(A): return True
    dif = [(a, b) for a, b in zip(A, B) if a != b]  # 这个比较两个string的方法很好
    return len(dif) == 2 and dif[0] == dif[1][::-1]


# 832. Flipping an Image
class Solution(object):
    def flipAndInvertImage(self, A):
        for row in A:
            for i in range((len(row) + 1) // 2):
                row[i], row[~i] = row[~i] ^ 1, row[i] ^ 1  # has to be binary
        return A
    # ~i 从-1开始，-1，-2，-3, a[~i]就是倒着从最后开始.


# 1. Two Sum to target value, return index
# 千万注意这里的list不是sorted， 所以不能一前一后的查找，必须两个循环找，下面的变形题可以。
class Solution:
    def twoSum(self, nums, target):

        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    return i, j


# 变形 167. Two Sum II - Input array is sorted
# Your returned answers (both index1 and index2) are not zero-based.
class Solution:
    """
    @param nums {int[]} n array of Integer
    @param target {int} = nums[index1] + nums[index2]
    @return {int[]} [index1 + 1, index2 + 1] (index1 < index2)
    """

    def twoSum(self, nums, target):
        # Write your code here
        l, r = 0, len(nums) - 1
        while l < r:
            value = nums[l] + nums[r]
            if value == target:
                return [l + 1, r + 1]
            elif value < target:
                l += 1
            else:
                r -= 1
        return []


# 13. Roman to integer
class Solution:
    # @param {string} s
    # @return {integer}
    def romanToInt(self, s):
        ROMAN = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }

        if s == "":
            return 0

        index = len(s) - 2
        sum = ROMAN[s[-1]]
        while index >= 0:
            if ROMAN[s[index]] < ROMAN[s[index + 1]]:
                sum -= ROMAN[s[index]]
            else:
                sum += ROMAN[s[index]]
            index -= 1
        return sum


# 12. Integer to Roman
M = ["", "M", "MM", "MMM"];
C = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"];
X = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"];
I = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"];
return M[num // 1000] + C[(num % 1000) // 100] + X[(num % 100) // 10] + I[num % 10];


# 344. Reverse String
class Solution:
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        return s[::-1]


# 541. Reverse String II
# Input: s = "abcdefg", k = 2
# Output: "bacdfeg"
def reverseStr(self, s, k):
    s = list(s)
    for i in xrange(0, len(s), 2 * k):
        s[i:i + k] = reversed(s[i:i + k])
    return "".join(s)


# 151. Given an input string, reverse the string word by word.
class Solution:
    # @param s : A string
    # @return : A string
    def reverseWords(self, s):
        return ' '.join(reversed(s.strip().split()))
        # 或者 return ' '.join(s.split()[::-1])
        # 或者 return " ".join(word for word in reversed(s.split(' ')))


# 落单的数 · Single Number
# 给出2*n + 1 个的数字，除其中一个数字之外其他每个数字均出现两次，找到这个数字。


# 14. Longest Common Prefix
class Solution:
    # @param strs: A list of strings
    # @return: The longest common prefix
    def longestCommonPrefix(self, strs):
        # write your code here
        if len(strs) <= 1:
            return strs[0] if len(strs) == 1 else ""
        end, minl = 0, min([len(s) for s in strs])
        while end < minl:
            for i in range(1, len(strs)):
                if strs[i][end] != strs[i - 1][end]:
                    return strs[0][:end]
            end = end + 1
        return strs[0][:end]


# 461. Hamming Distance
class Solution(Object)
    def hammingDistance(self, x, y)
        return ((bin(x ^ y)[2:]).count("1"))

# 477. Total Hamming Distance
??

# 9. Palindrome Number
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        x = str(x)
        return (x[::-1] == x)


# 变形Palindrome Number II 判断一个非负整数 n 的二进制表示是否为回文数
return str(bin(x)[2:])[::-1] == str(bin(x)[2:])  # didn't test


# 125. Valid Palindrome

def isPalindrome(self, s):
    l, r = 0, len(s) - 1
    while l < r:
        while l < r and not s[l].isalnum():
            l += 1
        while l < r and not s[r].isalnum():
            r -= 1
        if s[l].lower() != s[r].lower():
            return False
        l += 1;
        r -= 1
    return True


class Solution:
    # @param {string} s A string
    # @return {boolean} Whether the string is a valid palindrome
    def isPalindrome(self, s):
        start, end = 0, len(s) - 1
        while start < end:
            while start < end and not s[start].isalpha() and not s[start].isdigit():
                start += 1
            while start < end and not s[end].isalpha() and not s[end].isdigit():
                end -= 1
            if start < end and s[start].lower() != s[end].lower():
                return False
            start += 1
            end -= 1
        return True


# 409. Longest Palindrome
# Given a string which consists of lowercase or uppercase letters, find the length of
# the longest palindromes that can be built with those letters.
def longestPalindrome(self, s):
    odds = sum(v & 1 for v in collections.Counter(s).values())
    return len(s) - odds + bool(odds)


# 7. Reverse Integer
class Solution:
    # @param {int} n the integer to be reversed
    # @return {int} the reversed integer
    def reverseInteger(self, n):
        if n == 0:
            return 0

        neg = 1
        if n < 0:
            neg, n = -1, -n

        reverse = 0
        while n > 0:
            reverse = reverse * 10 + n % 10
            n = n / 10

        reverse = reverse * neg
        if reverse < -(1 << 31) or reverse > (1 << 31) - 1:
            return 0
        return reverse


# method2
class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        sign = [1, -1][x < 0]
        rst = sign * int(str(abs(x))[::-1])
        return rst if -(2 ** 31) - 1 < rst < 2 ** 31 else 0


# 变形 反转一个3位整数 · reverse 3 digit integer
class Solution:
    """
    @param number: A 3-digit integer
    @return: Reversed integer
    """

    def reverseInteger(self, number):
        return number % 10 * 100 + number / 10 % 10 * 10 + number / 100


# 21. Merge Two Sorted Lists
# Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.
class Solution(object):
    '''
    题意：合并两个有序链表
    '''

    def mergeTwoLists(self, l1, l2):
        dummy = ListNode(0)
        tmp = dummy
        while l1 != None and l2 != None:
            if l1.val < l2.val:
                tmp.next = l1
                l1 = l1.next
            else:
                tmp.next = l2
                l2 = l2.next
            tmp = tmp.next
        if l1 != None:
            tmp.next = l1
        else:
            tmp.next = l2
        return dummy.next


#  删除列表中节点 · Delete Node in a Linked List
#  这道题让我们删除链表的一个节点，更通常不同的是，没有给我们链表的起点，只给我们了一个要删的节点，
# 跟我们以前遇到的情况不太一样，我们之前要删除一个节点的方法是要有其前一个节点的位置，然后将其前一个
# 点的next连向要删节点的下一个，然后delete掉要删的节点即可。这道题的处理方法是先把当前节点的值用下
# 一个节点的值覆盖了，然后我们删除下一个节点即可

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val;
        node.next = node.next.next;


# 203. Remove Linked List Elements
# Remove all elements from a linked list of integers that have value val.


# 206. Reverse Linked List

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        curt = None
        while head != None:
            temp = head.next
            head.next = curt
            curt = head
            head = temp
        return curt


class Solution:
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        current = head
        while current != None:
            nxt = current.next
            current.next = prev
            prev = current
            current = nxt
        return prev

    # 92. Reverse Linked List II
    # Reverse a linked list from position m to n. Do it in one-pass.

    # # 561. Array Partition I
    # # https://docs.python.org/2.3/whatsnew/section-slices.html
    # class Solution:
    #     def arrayPairSum(self, nums):
    #         """
    #         :type nums: List[int]
    #         :rtype: int
    #         """
    #         return sum(sorted(nums)[::2])

    # 66 Plus one


class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        end = len(digits) - 1

        while end >= 0:
            if digits[end] < 9:
                digits[end] += 1
                return digits
            else:
                digits[end] = 0
                end -= 1
        else:
            return ([1] + digits)

** *  # 122. Best Time to Buy and Sell Stock II, no transactional fee 可以多次买卖
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/
# example [7,1,5,3,6,4]
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """

        profit = 0
        for i in range(len(prices) - 1):
            if prices[i] < prices[i + 1]:
                profit = profit + prices[i + 1] - prices[i]

        return profit;


# 121. Best Time to Buy and Sell Stock, 只能买卖一次, need to buy first
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        low = float('inf')
        profit = 0
        for i in range(len(prices)):
            if prices[i] < low:
                low = prices[i];
            elif prices[i] - low > profit:
                profit = prices[i] - low
        return profit


# 169. Majority Element

class Solution:
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        return nums[int(len(nums) / 2)]


# 283. Move Zeroes
class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        zero = 0  # records the position of "0"
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[zero] = nums[zero], nums[i]
                zero += 1


# 217. Contains Duplicate
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return len(nums) != len(set(nums))  # 不用len的时候 空list要出错

** *  # 219. Contains duplicate II

# 同样元素位置相差最大是k
# Given an array of integers and an integer k, find out whether there are two
# distinct indices i and j in the array such that nums[i] = nums[j] and the
# absolute difference between i and j is at most k.

def containsNearbyDuplicate(self, nums, k):
    dic = {}
    for i, v in enumerate(nums):
        if v in dic and i - dic[v] <= k:
            return True
        dic[v] = i
    return False


# 26. Remove Duplicates from Sorted Array
# 给定一个排序数组，在原数组中删除重复出现的数字，使得每个元素只出现一次，并且返回新的数组的长度。
# 不要使用额外的数组空间，必须在原地没有额外空间的条件下完成。


class Solution:
    """
    @param A: a list of integers
    @return an integer
    """

    def removeDuplicates(self, A):
        # write your code here
        if A == []:
            return 0
        index = 0
        for i in range(1, len(A)):
            if A[index] != A[i]:
                index += 1
                A[index] = A[i]

        return index + 1


# 27. Remove Element in place
# Given an array nums and a value val, remove all
# instances of that value in-place and return the new length
class Solution:
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        st, end = 0, len(nums) - 1

        while st <= end:
            if nums[st] == val:
                nums[st], nums[end] = nums[end], nums[st]
                end -= 1
            else:
                st += 1
        return st


# 88. merge sorted array in-place
# ou may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2.
class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        while m > 0 and n > 0:
            if nums1[m - 1] > nums2[n - 1]:
                nums1[m + n - 1] = nums1[m - 1]
                m -= 1
            else:
                nums1[m + n - 1] = nums2[n - 1]
                n -= 1
        nums1[:n] = nums2[:n]


# 104. Maximum Depth of Binary Tree
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        this.val = val
        this.left, this.right = None, None
"""


class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """

    def maxDepth(self, root):
        if root is None:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1


# 53. Maximum Subarray 异位动态规划
# Given an integer array nums, find the contiguous subarray (containing at least one number)
# which has the largest sum and return its sum
# [-2,1,-3,4,-1,2,1,-5,4]
class Solution:
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        maxSum = nums[0]
        curSum = nums[0]
        for i in range(1, len(nums)):
            curSum = max(curSum + nums[i], nums[i])  # 注意不是比较 curSum+num[i] 和 curSum
            if curSum > maxSum:
                maxSum = curSum
        return maxSum


class Solution:
    # @param A, a list of integers
    # @return an integer
    # 6:57
    def maxSubArray(self, A):
        if not A:
            return 0

        curSum = maxSum = A[0]
        for num in A[1:]:
            curSum = max(num, curSum + num)
            maxSum = max(maxSum, curSum)

        return maxSum


class Solution:

    def maxSubArray(self, nums):
        if nums is None or len(nums) == 0:
            return 0
        maxSum = nums[0]
        minSum = 0
        sum = 0
        for num in nums:
            sum += num
            if sum - minSum > maxSum:
                maxSum = sum - minSum
            if sum < minSum:
                minSum = sum
        return maxSum


# 643. Maximum Average Subarray I
# Given an array consisting of n integers, find the contiguous
# subarray of given length k that has the maximum average value.
# And you need to output the maximum average value.
def findMaxAverage(self, A, K):  #  注意这里k是给定的，有些题里长度不一定
    P = [0]
    for x in A:
        P.append(P[-1] + x)

    ma = max(P[i + K] - P[i] for i in range(len(A) - K + 1))
    return ma / float(K)


def findMaxAverage(self, nums, k):
    sums = [0] + list(itertools.accumulate(nums))
    return max(map(operator.sub, sums[k:], sums)) / k


def findMaxAverage(self, nums, k):
    sums = np.cumsum([0] + nums)
    return int(max(sums[k:] - sums[:-k])) / k


# 110. Balanced Binary Tree
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


class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """

    def isBalanced(self, root):
        balanced, _ = self.validate(root)
        return balanced

    def validate(self, root):
        if root is None:
            return True, 0

        balanced, leftHeight = self.validate(root.left)
        if not balanced:
            return False, 0
        balanced, rightHeight = self.validate(root.right)
        if not balanced:
            return False, 0

        return abs(leftHeight - rightHeight) <= 1, max(leftHeight, rightHeight) + 1


# fibonacci
class Solution:
    def Fibonacci(self, n):
        a = 0
        b = 1
        for i in range(n - 1):
            a, b = b, a + b
        return a


# 70. Climbing Stairs 类似fabinacci DP 问题
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


# 412. Fizz Buzz
class Solution:
    """
    @param n: An integer as description
    @return: A list of strings.
    For example, if n = 7, your code should return
        ["1", "2", "fizz", "4", "buzz", "fizz", "7"]
    """

    def fizzBuzz(self, n):
        results = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                results.append("fizz buzz")
            elif i % 5 == 0:
                results.append("buzz")
            elif i % 3 == 0:
                results.append("fizz")
            else:
                results.append(str(i))
        return results


def fizzBuzz(self, n):
    return ['Fizz' * (not i % 3) + 'Buzz' * (not i % 5) or str(i) for i in range(1, n + 1)]


#  258. Add Digits
class Solution(object):
    def addDigits(self, num):
        return num if num == 0 else num % 9 or 9


# 205. Isomorphic Strings
def isIsomorphic(self, s, t):
    return len(set(zip(s, t))) == len(set(s)) and len(set(zip(t, s))) == len(set(t))


def isIsomorphic1(self, s, t):
    d1, d2 = {}, {}
    for i, val in enumerate(s):
        d1[val] = d1.get(val, []) + [i]
    for i, val in enumerate(t):
        d2[val] = d2.get(val, []) + [i]
    return sorted(d1.values()) == sorted(d2.values())


def isIsomorphic2(self, s, t):
    d1, d2 = [[] for _ in xrange(256)], [[] for _ in xrange(256)]
    for i, val in enumerate(s):
        d1[ord(val)].append(i)
    for i, val in enumerate(t):
        d2[ord(val)].append(i)
    return sorted(d1) == sorted(d2)


def isIsomorphic3(self, s, t):
    return len(set(zip(s, t))) == len(set(s)) == len(set(t))


def isIsomorphic4(self, s, t):
    return [s.find(i) for i in s] == [t.find(j) for j in t]


def isIsomorphic5(self, s, t):
    return map(s.find, s) == map(t.find, t)


def isIsomorphic(self, s, t):
    d1, d2 = [0 for _ in xrange(256)], [0 for _ in xrange(256)]
    for i in xrange(len(s)):
        if d1[ord(s[i])] != d2[ord(t[i])]:
            return False
        d1[ord(s[i])] = i + 1
        d2[ord(t[i])] = i + 1
    return True


# 226. Invert Binary Tree (flip)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root:
            invert = self.invertTree
            root.left, root.right = invert(root.right), invert(root.left)
            return root


# 235. Lowest Common Ancestor of a Binary Search Tree (BST 一定是左比右边小)
class Solution:

    def lowestCommonAncestor(self, root, p, q):
        while root:
            if root.val > p.val and root.val > q.val:
                root = root.left
            elif root.val < p.val and root.val < q.val:
                root = root.right
            else:
                return root


# 669. Trim a Binary Search Tree
class Solution(object):
    def trimBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: TreeNode
        """
        if not root:
            return None
        if L > root.val:
            return self.trimBST(root.right, L, R)
        elif R < root.val:
            return self.trimBST(root.left, L, R)
        root.left = self.trimBST(root.left, L, R)
        root.right = self.trimBST(root.right, L, R)
        return root


class Solution:
    def trimBST(self, root, minimum, maximum):
        dummy = prev = TreeNode(0)

        while root != None:
            while root != None and root.val < minimum:
                root = root.right
            if root != None:
                prev.left = root
                prev = root
                root = root.left
        prev.left = None

        prev = dummy
        root = dummy.left
        while root != None:
            while root != None and root.val > maximum:
                root = root.left
            if root != None:
                prev.right = root
                prev = root
                root = root.right
        prev.right = None

        return dummy.right


# 538. Convert BST to Greater Tree
# Given a Binary Search Tree (BST), convert it to a Greater Tree such that every key
# of the original BST is changed to the original key plus sum of all keys greater than
# the original key in BST.
class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {TreeNode} the new root
    def convertBST(self, root):
        # Write your code here
        self.sum = 0
        self.helper(root)
        return root

    def helper(self, root):
        if root is None:
            return
        if root.right:
            self.helper(root.right)

        self.sum += root.val
        root.val = self.sum
        if root.left:
            self.helper(root.left)


# 236. Lowest Common Ancestor of a Binary Tree

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """

        if root in (None, p, q): return root
        left, right = (self.lowestCommonAncestor(kid, p, q)
                       for kid in (root.left, root.right))
        return root if left and right else left or right

    # 819. Most Common Word


def mostCommonWord(self, p, banned):
    ban = set(banned)
    words = re.sub(r'[^a-zA-Z]', ' ', p).lower().split()
    return collections.Counter(w for w in words if w not in ban).most_common(1)[0][0]


# 202. Happy Number
def isHappy(self, n):
    mem = set()
    while n != 1:
        n = sum([int(i) ** 2 for i in str(n)])
        if n in mem:
            return False
        else:
            mem.add(n)
    else:
        return True


def isHappy(self, n):
    seen = set()
    while n not in seen:
        seen.add(n)
        n = sum([int(x) ** 2 for x in str(n)])
    return n == 1


# 20. Valid Parentheses
class Solution(object):
    '''
    题意：输入一个只包含括号的字符串，判断括号是否匹配
    模拟堆栈，读到左括号压栈，读到右括号判断栈顶括号是否匹配
    '''

    def isValidParentheses(self, s):
        stack = []
        for ch in s:
            # 压栈
            if ch == '{' or ch == '[' or ch == '(':
                stack.append(ch)
            else:
                # 栈需非空
                if not stack:
                    return False
                # 判断栈顶是否匹配
                if ch == ']' and stack[-1] != '[' or ch == ')' and stack[-1] != '(' or ch == '}' and stack[-1] != '{':
                    return False
                # 弹栈
                stack.pop()
        return not stack


class Solution:
    # @return a boolean
    def isValid(self, s):
        stack = []
        dict = {"]": "[", "}": "{", ")": "("}
        for char in s:
            if char in dict.values():
                stack.append(char)
            elif char in dict.keys():
                if stack == [] or dict[char] != stack.pop():
                    return False
            else:
                return False
        return stack == []

** *  # binary search
# Binary search is a famous question in algorithm.
# For a given sorted array (ascending order) and a target number, find the first index of this number in O(log n) time complexity.
# If the target number does not exist in the array, return -1.
# Example If the array is [1, 2, 3, 3, 4, 5, 10], for given target 3, return 2.
class Solution:
    # @param nums: The integer array
    # @param target: Target number to find
    # @return the first position of target in nums, position start from 0
    def binarySearch(self, nums, target):

        if len(nums) == 0:
            return -1

        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = (start + end) / 2
            if nums[mid] < target:
                start = mid
            else:
                end = mid

        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        return -1

** *  # 74 search a 2D matrix
# Integers in each row are sorted from left to right.
# The first integer of each row is greater than the last integer of the previous row.
# 重点在于 下一行的第一个数大于前一行的最后一个数，其实整个matrix是个sorted
# return true or false
class Solution:
    """
    @param matrix, a list of lists of integers
    @param target, an integer
    @return a boolean, indicate whether matrix contains target
    """

    def searchMatrix(self, matrix, target):
        if len(matrix) == 0:
            return False

        n, m = len(matrix), len(matrix[0])
        start, end = 0, n * m - 1
        while start + 1 < end:
            mid = (start + end) / 2
            x, y = mid / m, mid % m
            if matrix[x][y] < target:
                start = mid
            else:
                end = mid
        x, y = start / m, start % m
        if matrix[x][y] == target:
            return True

        x, y = end / m, end % m
        if matrix[x][y] == target:
            return True

        return False


class Solution:
    # @param matrix, a list of lists of integers
    # @param target, an integer
    # @return a boolean
    def searchMatrix(self, matrix, target):
        if not matrix or target is None:
            return False

        rows, cols = len(matrix), len(matrix[0])
        low, high = 0, rows * cols - 1

        while low <= high:  # 这里必须有等于，否则结果不对
            mid = (low + high) // 2  # 必须要得到整数
            num = matrix[mid // cols][mid % cols]

            if num == target:
                return True
            elif num < target:
                low = mid + 1
            else:
                high = mid - 1

        return False


# 240. Search a 2D Matrix II
# 区别是并没有完全sorted
# 而是左右，上下sorted

class Solution:
    # @param {integer[][]} matrix
    # @param {integer} target
    # @return {boolean}
    def searchMatrix(self, matrix, target):
        if matrix:
            row, col, width = len(matrix) - 1, 0, len(matrix[0])
            while row >= 0 and col < width:
                if matrix[row][col] == target:
                    return True
                elif matrix[row][col] > target:
                    row = row - 1
                else:
                    col = col + 1
            return False


# 566. Reshape the Matrix
import numpy as np


class Solution(object):
    def matrixReshape(self, nums, r, c):
        try:
            return np.reshape(nums, (r, c)).tolist()
        except:
            return nums


def matrixReshape(self, nums, r, c):
    flat = sum(nums, [])
    if len(flat) != r * c:
        return nums
    tuples = zip(*([iter(flat)] * c))
    return map(list, tuples)


## 868. Transpose Matrix
class Solution(object):
    def transpose(self, A):
        return list(zip(*A))  # without list, it returns an iterator


# A = [[1,2,3],[4,5,6],[7,8,9]]
# *A 成了 [1, 2, 3] [4, 5, 6] [7, 8, 9]
# zip 就是每个list第一个元素combine起来 [(1, 4, 7), (2, 5, 8), (3, 6, 9)]


class Solution:
    def transpose(self, A):
        return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


# 204. Count Primes Count the number of prime numbers less than a non-negative number, n.
class Solution:


# @param {integer} n
# @return {integer}
def countPrimes(self, n):
    if n < 3:
        return 0
    primes = [True] * n
    primes[0] = primes[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if primes[i]:
            primes[i * i: n: i] = [False] * len(primes[i * i: n: i])
    return sum(primes)

    def countPrimes(self, n):
        if n <= 2:
            return 0

    res = [True] * n
    res[0] = res[1] = False
    for i in range(2, n):
        if res[i] == True:
            for j in range(2, (n - 1) // i + 1):
                res[i * j] = False
    return sum(res)

    # 172. Factorial Trailing Zeroes
    def trailingZeroes(self, n):
        r = 0
        while n > 0:
            n /= 5
            r += n
        return r


# 617. Merge Two Binary Trees
def mergeTrees(self, t1, t2):
    if not t1 and not t2: return None
    ans = TreeNode((t1.val if t1 else 0) + (t2.val if t2 else 0))
    ans.left = self.mergeTrees(t1 and t1.left, t2 and t2.left)
    ans.right = self.mergeTrees(t1 and t1.right, t2 and t2.right)
    return ans


# 530. Minimum Absolute Difference in BST
def getMinimumDifference(self, root):
    def dfs(node, l=[]):
        if node.left: dfs(node.left, l)
        l.append(node.val)
        if node.right: dfs(node.right, l)
        return l

    l = dfs(root)
    return min([abs(a - b) for a, b in zip(l, l[1:])])


# 2. Add Two Numbers
# You are given two non-empty linked lists representing two non-negative integers.
# The digits are stored in reverse order and each of their nodes contain a single digit.
# Add the two numbers and return it as a linked list.
class Solution:
    # @return a ListNode
    def addTwoNumbers(self, l1, l2):
        carry = 0
        root = n = ListNode(0)
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1 + v2 + carry, 10)
            n.next = ListNode(val)
            n = n.next
        return root.next


# 118. Pascal's triangle
# Given a non-negative integer numRows, generate the first numRows of Pascal's triangle.
# input 5, generate all first 5 rows with correct numbers
def generate(numRows):
    a = [[1] * (i + 1) for i in range(numRows)]
    for i in range(2, numRows):
        for j in range(1, i):
            a[i][j] = a[i - 1][j - 1] + a[i - 1][j]
    return a


# def generate(self, numRows):
#         res = [[1]]
#         for i in range(1, numRows):
#             res += [map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1])]
#         return res[:numRows]


# 119. Pascal's Triangle II
# Given a non-negative index k where k ≤ 33, return the kth index row of the Pascal's triangle.
class Solution:
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """

        pascal = [[1] * (i + 1) for i in range(rowIndex + 1)]
        for i in range(2, rowIndex + 1):
            for j in range(1, i):
                pascal[i][j] = pascal[i - 1][j - 1] + pascal[i - 1][j]
        return pascal[rowIndex]

    # zip没看懂
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        row = [1]
        for _ in range(rowIndex):
            row = [x + y for x, y in zip([0] + row, row + [0])]
        return row


# 268. Missing Number
# Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.
def missingNumber(self, nums):
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)


# 448. Find All Numbers Disappeared in an Array
# Given an array of integers where 1 ≤ a[i] ≤ n
# (n = size of array), some elements appear twice and others appear once.

Find
all
the
elements
of[1, n]
inclusive
that
do
not appear in this
array.


class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # For each number i in nums,
        # we mark the number that i points as negative.
        # Then we filter the list, get all the indexes
        # who points to a positive number
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = - abs(nums[index])

        return [i + 1 for i in range(len(nums)) if nums[i] > 0]


def findDisappearedNumbers(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    return list(set(range(1, len(nums) + 1)) - set(nums))  # 注意list这里是圆括号


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

    # method 2
    r = x
    while r * r > x:
        r = (r + x // r) // 2
    return r


# 697. Degree of an Array
# Given a non-empty array of non-negative integers nums, the degree
# of this array is defined as the maximum frequency of any one of its
# elements.
# Your task is to find the smallest possible length of a (contiguous)
# subarray of nums, that has the same degree as nums.

def findShortestSubArray(self, nums):
    map = defaultdict(list)
    for i in range(len(nums)):
        map[nums[i]].append(i)
    return min((-len(list), list[-1] - list[0] + 1) for list in map.values())[1]


# good methods to sort out elements in a list and their position


def findShortestSubArray(self, nums):
    first, last = {}, {}
    for i, v in enumerate(nums):
        first.setdefault(v, i)
        last[v] = i
    c = collections.Counter(nums)
    degree = max(c.values())
    return min(last[v] - first[v] + 1 for v in c if c[v] == degree)


class Solution(object):
    def findShortestSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        c = Counter(nums)
        degree = max(c.values())
        m = defaultdict(list)
        for i in range(len(nums)):
            m[nums[i]].append(i)
        return min(m[k][-1] - m[k][0] + 1 for k in m.keys() if c[k] == degree)


def findShortestSubArray(self, nums):
    map = defaultdict(list)
    for i in range(len(nums)):
        map[nums[i]].append(i)
    return min((-len(list), list[-1] - list[0] + 1) for list in map.values())[1]


# 189. Rotate Array

class Solution:
    # @param nums, a list of integer
    # @param k, num of steps
    # @return nothing, please modify the nums list in-place.
    def rotate(self, nums, k):
        n = len(nums)
        # k = k % n
        nums[:] = nums[n - k:] + nums[:n - k]  # be careful nums[:]
        #  要考虑的是k=0


# find common elements in 2 list

# 349. Intersection of Two Arrays
# Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2]
# 只用找出有交集的元素
class Solution:
    # @param {int[]} nums1 an integer array
    # @param {int[]} nums2 an integer array
    # @return {int[]} an integer array
    def intersection(self, nums1, nums2):
        # Write your code here
        return list(set(nums1) & set(nums2))


# 350. Intersection of Two Arrays II
# Each element in the result should appear as many times as it shows in both arrays.

def intersect(self, nums1, nums2):
    a, b = map(collections.Counter, (nums1, nums2))
    return list((a & b).elements())


# nums1 = [1,2,2,1]
# nums2 = [2,2]
# a, b = map(collections.Counter, (nums1, nums2))
# (Counter({1: 2, 2: 2}), Counter({2: 2}))


# 682. Baseball Game
class Solution(object):
    def calPoints(self, ops):
        # Time: O(n)
        # Space: O(n)
        history = []
        for op in ops:
            if op == 'C':
                history.pop()
            elif op == 'D':
                history.append(history[-1] * 2)
            elif op == '+':
                history.append(history[-1] + history[-2])
            else:
                history.append(int(op))
        return sum(history)


# newton method
def dx(f, x):
    return abs(0 - f(x))


def newtons_method(f, df, x0, e):
    delta = dx(f, x0)
    while delta > e:
        x0 = x0 - f(x0) / df(x0)
        delta = dx(f, x0)
    print
    'Root is at: ', x0
    print
    'f(x) at root is: ', f(x0)






