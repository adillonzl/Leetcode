"""
206 Reverse Linked List
92	Reverse Linked List II
141	Linked List Cycle
24	Swap Nodes in Pairs
328	Odd Even Linked List

237	Delete Node in a Linked List
19	Remove Nth Node From End of List
83	Remove Duplicates from Sorted List
203	Remove Linked List Elements
82	Remove Duplicates from Sorted List II
369	Plus One Linked List
2	Add Two Numbers
160	Intersection of Two Linked Lists
21	Merge Two Sorted Lists
提高
234	Palindrome Linked List
143	Reorder List
142	Linked List Cycle II
148	Sort List
25	Reverse Nodes in k-Group
61	Rotate List
86	Partition List
23	Merge k Sorted Lists
147	Insertion Sort List
"""
"""
# 206 Reverse Linked List
Reverse a singly linked list.

Example:

Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
Follow up:

A linked list can be reversed either iteratively or recursively. 
Could you implement both?"""

"""
https://leetcode.com/problems/reverse-linked-list/discuss/140916/Python-Iterative-and-Recursive-(206)

Iterative: 设置一个Prev的参数方便之后的操作。

Step
1: 保存head reference
在While循环中，cur指针用来保留当前Head的位置，因为如果我们操作
head = head.next这一步并且没有对head的位置进行保留， 我们会失去对head的reference，
导致我们之后不能进行反转操作。

Step
2: 保存head.next的reference
head的reference存好以后，我们还需要保存head.next的reference，原因是我们如果对第一个node进行了反转操作，node就指向我们之前定义好的prev上，而失去了对原先head.next这个位置的拥有权。

head.next这个reference，我们直接用head来保存即可，所以有了head = head.next这么一个操作。当然你要是想写的更加易懂，你也可以直接新创建一个函数，取名next，然后指向next = head.next。

Step3: 反转. 万事俱备，可以对cur指针进行反转了，指向之前定义的prev。

Step 4: Reference转移 . 最后不要忘记移动prev到cur的位置，不然prev的位置永远不变
"""

class Solution:
    def reverseList(self, head):
        prev = None
        while head:
            cur = head
            head = head.next
            cur.next = prev
            prev = cur
        return prev

"""
Iterative(保持Head位置)
底下这种写法，保证了Head的位置不被移动，这种操作对于该题的定义毫无意义，因为最终不需要head的reference，但如果API定义需要head的话，是个好的practice.
还有就是对prev和next两个指针的命名，提高了code的可读性。
"""
class Solution(object):
    def reverseList(self, head):
        if not head: return None
        prev, cur = None, head

        while cur:
            next = cur.next
            cur.next = prev
            prev = cur
            cur = next
        return prev


#Recursive
class Solution:
    def reverseList(self, head):
        if not head or not head.next:
            return head

        new_head = self.reverseList(head.next)
        next_node = head.next  # head -> next_node 
        next_node.next = head  # head <- next_node 
        head.next = None  # [x] <- head <- next_node 
        return new_head
"""
Step1: Base Case: if not head or not head.next:
    return head
Base Case的返回条件是当head或者head.next为空，则开始返回

Step 2: Recurse到底
new_head = self.reverseList(head.next)
在做任何处理之前，我们需要不断的递归，直到触及到Base
Case,。当我们移动到最后一个Node以后, 将这个Node定义为我们新的head, 取名new_head.我们从new_head开始重复性的往前更改指针的方向

Step
3: 返回并且更改指针方向
next_node = head.next  # head -> next_node 
next_node.next = head  # head <- next_node 
head.next = None  # [x] <- head <- next_node 

这里一定要注意了，我们每次往上返回的是new_head, 这个new_head指针指向的是最后的Node.而我们再返回过程中实际操作的head，是在step2: Recurse到底这一步向下传递的head的指针, 不要搞混了, 实在不懂，就把上面这个图画一下，走一遍

Follow up: Reverse Nodes in K - Group
利用上面这种保持Head位置的写法，可以直接将方程带入到这道题常问的Follow
Up 也就是Leetcode 25
的原题，代码也放在这

含义就是我们找到每次新的head的位置，然后通过我们这题写好的逻辑翻转即可。
"""
class Solution(object):
    def reverseKGroup(self, head, k):
        count, node = 0, head
        while node and count < k:
            node = node.next
            count += 1
        if count < k: return head
        new_head, prev = self.reverse(head, count)
        head.next = self.reverseKGroup(new_head, k)
        return prev

    def reverse(self, head, count):
        prev, cur, nxt = None, head, head
        while count > 0:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
            count -= 1
        return (cur, prev)

# 92. Reverse Linked List II
"""
Reverse a linked list from position m to n. Do it in one-pass.

Note: 1 ≤ m ≤ n ≤ length of list.

Example:

Input: 1->2->3->4->5->NULL, m = 2, n = 4
Output: 1->4->3->2->5->NULL
"""
#The idea is simple and intuitive: find linkedlist [m, n], reverse it, then connect m with n+1, connect n with m-1
class Solution:
    # @param head, a ListNode
    # @param m, an integer
    # @param n, an integer
    # @return a ListNode
    def reverseBetween(self, head, m, n):
        if m == n:
            return head

        dummyNode = ListNode(0)
        dummyNode.next = head
        pre = dummyNode

        for i in range(m - 1):
            pre = pre.next

        # reverse the [m, n] nodes
        reverse = None
        cur = pre.next
        for i in range(n - m + 1):
            next = cur.next
            cur.next = reverse
            reverse = cur
            cur = next

        pre.next.next = cur
        pre.next = reverse

        return dummyNode.next

# 141	Linked List Cycle
"""
Given a linked list, determine if it has a cycle in it.

To represent a cycle in the given linked list, we use an integer pos which represents 
the position (0-indexed) in the linked list where tail connects to. If pos is -1, 
then there is no cycle in the linked list.
Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where tail connects to the second node.

"""
# 使用额外空间
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head or not head.next:
            return False
        nodes_seen = set()
        curr = head
        while curr:
            if curr not in nodes_seen:
                nodes_seen.add(curr)
                curr = curr.next
            else:
                return True
        return False

# cycle detection 龟兔赛跑
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head or not head.next:
            return False
        bunny, turtle = head.next, head
        while bunny != turtle:
            if not bunny.next or not bunny.next.next:
                return False
            bunny = bunny.next.next
            turtle = turtle.next
        return True

def hasCycle(self, head):
    try:
        slow = head
        fast = head.next
        while slow is not fast:
            slow = slow.next
            fast = fast.next.next
        return True
    except:
        return False

#24	Swap Nodes in Pairs
"""
Given a linked list, swap every two adjacent nodes and return its head.

You may not modify the values in the list's nodes, only nodes itself may be changed.

Example:

Given 1->2->3->4, you should return the list as 2->1->4->3.
"""
"""
Here, pre is the previous node. Since the head doesn't have a previous node, 
I just use self instead. Again, a is the current node and b is the next node.

To go from pre -> a -> b -> b.next to pre -> b -> a -> b.next, we need to change 
those three references. Instead of thinking about in what order I change them, 
I just change all three at once.
"""
def swapPairs(self, head):
    pre, pre.next = self, head
    while pre.next and pre.next.next:
        a = pre.next
        b = a.next
        pre.next, b.next, a.next = b, a, b.next
        pre = a
    return self.next

# 328	Odd Even Linked List
"""
Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes.

You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time complexity.

Example 1:

Input: 1->2->3->4->5->NULL
Output: 1->3->5->2->4->NULL
Example 2:

Input: 2->1->3->5->6->4->7->NULL
Output: 2->3->6->7->1->5->4->NULL

"""
def oddEvenList(self, head):
    dummy1 = odd = ListNode(0)
    dummy2 = even = ListNode(0)
    while head:
        odd.next = head
        even.next = head.next
        odd = odd.next
        even = even.next
        head = head.next.next if even else None
    odd.next = dummy2.next
    return dummy1.next

class Solution:
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        even = ListNode(-9999)
        even_start = even

        odd = ListNode(-9999)
        odd_start = odd

        current_odd = True

        while head:
            if current_odd:
                odd.next = head
                odd = odd.next
            else:
                even.next = head
                even = even.next

            head = head.next
            current_odd = not current_odd

        even.next = None
        odd.next = even_start.next
        return odd_start.next

# 237	Delete Node in a Linked List
"""
Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.

Given linked list -- head = [4,5,1,9], which looks like following:

Example 1:

Input: head = [4,5,1,9], node = 5
Output: [4,1,9]
Explanation: You are given the second node with value 5, the linked list should 
become 4 -> 1 -> 9 after calling your function."""
def deleteNode(self, node):
    node.val = node.next.val
    node.next = node.next.next


#19	Remove Nth Node From End of List
"""
Given a linked list, remove the n-th node from the end of list and return its head.

Example:

Given linked list: 1->2->3->4->5, and n = 2.

After removing the second node from the end, the linked list becomes 1->2->3->5.
Note: Given n will always be valid.

Follow up: Could you do this in one pass?"""
#Value-Shifting - AC in 64 ms

#My first solution is "cheating" a little. Instead of really removing the nth node,
# I remove the nth value. I recursively determine the indexes (counting from back),
# then shift the values for all indexes larger than n, and then always drop the head.

class Solution:
    def removeNthFromEnd(self, head, n):
        def index(node):
            if not node:
                return 0
            i = index(node.next) + 1
            if i > n:
                node.next.val = node.val
            return i
        index(head)
        return head.next

#Index and Remove - AC in 56 ms

#In this solution I recursively determine the indexes again, but this time my helper
# function removes the nth node. It returns two values. The index, as in my first solution,
# and the possibly changed head of the remaining list.

class Solution:
    def removeNthFromEnd(self, head, n):
        def remove(head):
            if not head:
                return 0, head
            i, head.next = remove(head.next)
            return i+1, (head, head.next)[i+1 == n]
        return remove(head)[1]

#n ahead - AC in 48 ms

# The standard solution, but without a dummy extra node. Instead, I simply handle the
# special case of removing the head right after the fast cursor got its head start.

class Solution:
    def removeNthFromEnd(self, head, n):
        fast = slow = head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head

#83	Remove Duplicates from Sorted List
"""
Given a sorted linked list, delete all duplicates such that each element appear only once.

Example 1:

Input: 1->1->2
Output: 1->2
Example 2:

Input: 1->1->2->3->3
Output: 1->2->3
"""
def deleteDuplicates(self, head):
    if head and head.next:
        head.next = self.deleteDuplicates(head.next)
        return head.next if head.next.val == head.val else head
    return head

def deleteDuplicates(self, head):
    cur = head
    while cur:
        while cur.next and cur.next.val == cur.val:
            cur.next = cur.next.next     # skip duplicated node
        cur = cur.next     # not duplicate of current node, move to next node
    return head


#82	Remove Duplicates from Sorted List II
"""Given a sorted linked list, delete all nodes that have duplicate numbers, 
leaving only distinct numbers from the original list.

Example 1:

Input: 1->2->3->3->4->4->5
Output: 1->2->5
Example 2:

Input: 1->1->1->2->3
Output: 2->3
"""
def deleteDuplicates(self, head):
    dummy = pre = ListNode(0)
    dummy.next = head
    while head and head.next:
        if head.val == head.next.val:
            while head and head.next and head.val == head.next.val:
                head = head.next
            head = head.next
            pre.next = head
        else:
            pre = pre.next
            head = head.next
    return dummy.next


def deleteDuplicates(self, head):
    # Add a dummy node point to the current list
    newhead = ListNode(0)
    newhead.next = head
    val_need_to_be_deleted = None
    tail = newhead

    while head:

        # Triger delete mode if current has the same val as the next
        if head and head.next and head.val == head.next.val:
            val_need_to_be_deleted = head.val

        # Not a dup if delete mode is off or the current value doesn't match the value need to be deleted
        if val_need_to_be_deleted == None or head.val != val_need_to_be_deleted:
            # add it to the newlist
            tail.next = head
            tail = head

        head = head.next

    tail.next = None
    return newhead.next


#203 Remove Linked List Elements
"""
Remove all elements from a linked list of integers that have value val.

Example:

Input:  1->2->6->3->4->5->6, val = 6
Output: 1->2->3->4->5"""

"""
Before writing any code, it's good to make a list of edge cases that we need to consider. This is so that we can be certain that we're not overlooking anything while coming up with our algorithm, and that we're testing all special cases when we're ready to test. These are the edge cases that I came up with.

The linked list is empty, i.e. the head node is None.
Multiple nodes with the target value in a row.
The head node has the target value.
The head node, and any number of nodes immediately after it have the target value.
All of the nodes have the target value.
The last node has the target value.
So with that, this is the algorithm I came up with.

"""
class Solution:
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """

        dummy_head = ListNode(-1)
        dummy_head.next = head

        current_node = dummy_head
        while current_node.next != None:
            if current_node.next.val == val:
                current_node.next = current_node.next.next
            else:
                current_node = current_node.next

        return dummy_head.next


"""In order to save the need to treat the "head" as special, the algorithm uses a "dummy" 
head. This simplifies the code greatly, particularly in the case of needing to remove the 
head AND some of the nodes immediately after it.

Then, we keep track of the current node we're up to, and look ahead to its next node, as 
long as it exists. If current_node.next does need removing, then we simply replace it with 
current_node.next.next. We know this is always "safe", because current_node.next is 
definitely not None (the loop condition ensures that), so we can safely access its next.

Otherwise, we know that current_node.next should be kept, and so we move current_node on 
to be current_node.next.

The loop condition only needs to check that current_node.next != None. The reason it does not need to check that current_node != None is because this is an impossible state to reach. Think about it this way: The ONLY case that we ever do current_node = current_node.next in is immediately after the loop has already confirmed that current_node.next is not None.

The algorithm requires O(1) extra space and takes O(n) time.
"""


#locked 369	Plus One Linked List
#2	Add Two Numbers
"""You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807."""
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
        carry, val = divmod(v1+v2+carry, 10)
        n.next = ListNode(val)
        n = n.next
    return root.next


def addTwoNumbers(self, l1, l2):
    dummy = cur = ListNode(0)
    carry = 0
    while l1 or l2 or carry:
        if l1:
            carry += l1.val
            l1 = l1.next
        if l2:
            carry += l2.val
            l2 = l2.next
        cur.next = ListNode(carry%10)
        cur = cur.next
        carry //= 10
    return dummy.next


#160	Intersection of Two Linked Lists
"""https://leetcode.com/problems/intersection-of-two-linked-lists/"""
class Solution:
    # @param two ListNodes
    # @return the intersected ListNode
    def getIntersectionNode(self, headA, headB):
        if headA is None or headB is None:
            return None

        pa = headA # 2 pointers
        pb = headB

        while pa is not pb:
            # if either pointer hits the end, switch head and continue the second traversal,
            # if not hit the end, just move on to next
            pa = headB if pa is None else pa.next
            pb = headA if pb is None else pb.next

        return pa # only 2 ways to get out of the loop, they meet or the both hit the end=None

# the idea is if you switch head, the possible difference between length would be countered.
# On the second traversal, they either hit or miss.
# if they meet, pa or pb would be the node we are looking for,
# if they didn't meet, they will hit the end at the same iteration, pa == pb == None, return
# either one of them is the same,None

#method2
class Solution:
    # @param two ListNodes
    # @return the intersected ListNode
    def getIntersectionNode(self, headA, headB):
        curA,curB = headA,headB
        lenA,lenB = 0,0
        while curA is not None:
            lenA += 1
            curA = curA.next
        while curB is not None:
            lenB += 1
            curB = curB.next
        curA,curB = headA,headB
        if lenA > lenB:
            for i in range(lenA-lenB):
                curA = curA.next
        elif lenB > lenA:
            for i in range(lenB-lenA):
                curB = curB.next
        while curB != curA:
            curB = curB.next
            curA = curA.next
        return curA
"""The solution is straightforward: maintaining two pointers in the lists under the 
constraint that both lists have the same number of nodes starting from the pointers. 
We need to calculate the length of each list though. So O(N) for time and O(1) for space."""

#21	Merge Two Sorted Lists
"""Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.

Example:

Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4"""


# iteratively
def mergeTwoLists1(self, l1, l2):
    dummy = cur = ListNode(0)
    while l1 and l2:
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next


# recursively
def mergeTwoLists2(self, l1, l2):
    if not l1 or not l2:
        return l1 or l2
    if l1.val < l2.val:
        l1.next = self.mergeTwoLists(l1.next, l2)
        return l1
    else:
        l2.next = self.mergeTwoLists(l1, l2.next)
        return l2


# in-place, iteratively
def mergeTwoLists(self, l1, l2):
    if None in (l1, l2):
        return l1 or l2
    dummy = cur = ListNode(0)
    dummy.next = l1
    while l1 and l2:
        if l1.val < l2.val:
            l1 = l1.next
        else:
            nxt = cur.next
            cur.next = l2
            tmp = l2.next
            l2.next = nxt
            l2 = tmp
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next

#23	Merge k Sorted Lists
"""Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

Example:

Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6"""
from Queue import PriorityQueue
class Solution(object):
    def mergeKLists(self, lists):
        dummy = ListNode(None)
        curr = dummy
        q = PriorityQueue()
        for node in lists:
            if node: q.put((node.val,node))
        while q.qsize()>0:
            curr.next = q.get()[1]
            curr=curr.next
            if curr.next: q.put((curr.next.val, curr.next))
        return dummy.next

def mergeKLists(self, lists):
    from heapq import heappush, heappop, heapreplace, heapify
    dummy = node = ListNode(0)
    h = [(n.val, n) for n in lists if n]
    heapify(h)
    while h:
        v, n = h[0]
        if n.next is None:
            heappop(h)  # only change heap size when necessary
        else:
            heapreplace(h, (n.next.val, n.next))
        node.next = n
        node = node.next

    return dummy.next