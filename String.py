'''
基础
28	Implement strStr()
14	Longest Common Prefix
58	Length of Last Word
387	First Unique Character in a String
383	Ransom Note
344	Reverse String
151	Reverse Words in a String
186	Reverse Words in a String II
345	Reverse Vowels of a String
205	Isomorphic Strings
293	Flip Game
294	Flip Game II
290	Word Pattern
242	Valid Anagram
49	Group Anagrams
249	Group Shifted Strings
87	Scramble String
179	Largest Number	很少考
6	ZigZag Conversion	很少考
161	One Edit Distance
38	Count and Say
358	Rearrange String k Distance Apart
316	Remove Duplicate Letters
271	Encode and Decode Strings
168	Excel Sheet Column Title
171	Excel Sheet Column Number
13	Roman to Integer
12	Integer to Roman
273	Integer to English Words
246	Strobogrammatic Number
247	Strobogrammatic Number II
248	Strobogrammatic Number III	很少考
提高
68	Text Justification
65	Valid Number
157	Read N Characters Given Read4
158	Read N Characters Given Read4 II - Call multiple times
Substring
76	Minimum Window Substring	Sliding Window
30	Substring with Concatenation of All Words	Sliding Window
3	Longest Substring Without Repeating Characters	Sliding Window
340	Longest Substring with At Most K Distinct Characters	Sliding Window
395	Longest Substring with At Least K Repeating Characters	Sliding Window
159	Longest Substring with At Most Two Distinct Characters	Sliding Window
Palindrome
125	Valid Palindrome
266	Palindrome Permutation
5	Longest Palindromic Substring
9	Palindrome Number
214	Shortest Palindrome
336	Palindrome Pairs
131	Palindrome Partitioning
132	Palindrome Partitioning II
267	Palindrome Permutation II
Parentheses
20	Valid Parentheses
22	Generate Parentheses
32	Longest Valid Parentheses
241	Different Ways to Add Parentheses
301	Remove Invalid Parentheses
Subsequence
392	Is Subsequence
115	Distinct Subsequences
187	Repeated DNA Sequences	很少考
'''
# 1108. Defanging an IP Address
class Solution(object):
    def defangIPaddr(self, address):
        """
        :type address: str
        :rtype: str
        """

        return address.replace('.', '[.]')

# 709. To Lower Case
class Solution:
    def toLowerCase(self, str):
        return "".join(chr(ord(c) + 32) if 65 <= ord(c) <= 90 else c for c in str)

    # c is unmutable so, cannot assign directly


class Solution(object):
    def isUpperChar(self, c):
        if c >= 'A' and c <= 'Z':
            return True
        return False

    def toLowerCase(self, str):
        """
        :type str: str
        :rtype: str
        """
        s = list(str)
        for i in range(0, len(s)):
            char = s[i]
            if 65 <= ord(char) <= 90:
                s[i] = chr(ord(char) + 32)

        return "".join(s)


class Solution:
    def toLowerCase(self, str):
        return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in str)
class Solution:
    def toLowerCase(self, str):
        return str.lower()

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

class Solution(object):
    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        ab=[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        temp=[]
        for item in set(words):
            s=''
            for ch in item:
                s+=ab[ord(ch)-97]
            temp.append(s)
        return len(set(temp))

class Solution(object):
    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        morse = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        alphabet = list(map(chr, range(97, 123)))
        d = {alphabet[i]: morse[i] for i in range(len(morse))}
        s = set()
        for w in words:
            s.add(''.join([d[ch] for ch in w]))
        return len(s)



# 657. Robot Return to Origin
def judgeCircle(self, moves):
    return moves.count('L') == moves.count('R') and moves.count('U') == moves.count('D')


class Solution(object):
    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """
        rCount = 0
        uCount = 0
        for char in moves:
            if char == 'U':
                uCount = uCount + 1
            elif char == 'R':
                rCount = rCount + 1
            elif char == 'D':
                uCount = uCount - 1
            else:
                rCount = rCount - 1
        if uCount == 0 and rCount == 0:
            return True
        return False


# 929. Unique Email Addresses
class Solution:
    def numUniqueEmails(self, emails):
        """
        :type emails: List[str]
        :rtype: int
        """
        email_set = set()
        for email in emails:
            local_name,domain_name = email.split("@")
            local_name ="".join(local_name.split('+')[0].split('.'))
            email = local_name +'@' + domain_name
            email_set.add(email)
        return len(email_set)


class Solution(object):
    def numUniqueEmails(self, emails):
        """
        :type emails: List[str]
        :rtype: int
        """
        set_emails = set()
        for email in emails:
            local, domain = email.split("@")
            while "." in local:
                local = local.replace(".", "")
            if "+" in local:
                index = local.index("+")
                local = local[:index]
            reformatted = local + "@" + domain
            set_emails.add(reformatted)
        return len(set_emails)

# 344. Reverse String
"""
Write a function that reverses a string. The input string is given as an array of characters char[].

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

You may assume all the characters consist of printable ascii characters.
Example 1:

Input: ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]
Example 2:

Input: ["H","a","n","n","a","h"]
Output: ["h","a","n","n","a","H"]
"""
# 3 solutions: Recursive, Classic, Pythonic
class Solution(object):
    def reverseString(self, s):
        l = len(s)
        if l < 2:
            return s
        return self.reverseString(s[l/2:]) + self.reverseString(s[:l/2])


class SolutionClassic(object):
    def reverseString(self, s):
        r = list(s)
        i, j  = 0, len(r) - 1
        while i < j:
            r[i], r[j] = r[j], r[i]
            i += 1
            j -= 1

        return "".join(r)

class SolutionPythonic(object):
    def reverseString(self, s):
        return s[::-1]


# 541. Reverse String II
"""
Given a string and an integer k, you need to reverse the first k characters for every 2k characters counting from the start of the string. If there are less than k characters left, reverse all of them. If there are less than 2k but greater than or equal to k characters, then reverse the first k characters and left the other as original.
Example:
Input: s = "abcdefg", k = 2
Output: "bacdfeg"
"""
def reverseStr(self, s, k):
    s = list(s)
    for i in xrange(0, len(s), 2*k):
        s[i:i+k] = reversed(s[i:i+k])
    return "".join(s)


class Solution(object):
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """

        r = []
        len_s = len(s)
        for i in range(0, len_s, 2 * k):
            start = i
            mid = i + k
            if mid > len_s:
                mid = len_s

            end = i + 2 * k
            if end > len_s:
                end = len_s

            first_k = s[start:mid]
            r.append(first_k[::-1])
            if (end > mid):
                r.append(s[mid:end])

        return ''.join(r)


# 557. Reverse Words in a String III
"""Given a string, you need to reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order.

Example 1:
Input: "Let's take LeetCode contest"
Output: "s'teL ekat edoCteeL tsetnoc"
Note: In the string, each word is separated by single space and there will not be any extra space in the string.
"""
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        word = s.split()
        new = ' '.join(x[::-1] for x in word)
        return new

class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        return " ".join(map(lambda x: x[::-1], s.split()))

#Here I first reverse the order of the words and then reverse the entire string.

def reverseWords(self, s):
    return ' '.join(s.split()[::-1])[::-1]
# That's a bit shorter than the more obvious one:

def reverseWords(self, s):
    return ' '.join(x[::-1] for x in s.split())

# 893. Groups of Special-Equivalent Strings
class Solution:
    def numSpecialEquivGroups(self, A):
        return len(set("".join(sorted(s[0::2])) + "".join(sorted(s[1::2])) for s in A))

# 1170. Compare Strings by Frequency of the Smallest Character


# 917. Reverse Only Letters
"""
Given a string S, return the "reversed" string where all characters that are not a letter stay in the same place, and all letters reverse their positions.

 

Example 1:

Input: "ab-cd"
Output: "dc-ba"
Example 2:

Input: "a-bC-dEf-ghIj"
Output: "j-Ih-gfE-dCba"
Example 3:

Input: "Test1ng-Leet=code-Q!"
Output: "Qedo1ct-eeLg=ntse-T!"
"""
class Solution(object):
    def reverseOnlyLetters(self, S):
        """
        :type S: str
        :rtype: str
        """
        low, high =0, len(S)-1
        s = list(S)
        while low<high:
            if s[low].isalpha() == False:
                low +=1
            elif s[high].isalpha() == False:
                high-=1
            else:
                s[low],s[high] = s[high], s[low]
                low+=1
                high-=1
        return ''.join(s)


class Solution(object):
    def reverseOnlyLetters(self, s):
        """
        :type S: str
        :rtype: str
        """

        start = 0
        end = len(s) - 1
        s = list(s)

        while (start < end and start >= 0 and end < len(s)):
            print
            start
            print
            end
            if (s[start].isalpha() and s[end].isalpha()):
                temp = s[start]
                s[start] = s[end]
                s[end] = temp
                start = start + 1
                end = end - 1

            elif (not s[start].isalpha()):
                start = start + 1
            elif (not s[end].isalpha()):
                end = end - 1

        return "".join(s)


class Solution(object):
    def reverseOnlyLetters(self, S):
        """
        :type S: str
        :rtype: str
        """

        S, i, j = list(S), 0, len(S) - 1
        while i < j:
            if not S[i].isalpha():
                i += 1
            elif not S[j].isalpha():
                j -= 1
            else:
                S[i], S[j] = S[j], S[i]
                i, j = i + 1, j - 1
        return "".join(S)

# 937. Reorder Data in Log Files
"""
You have an array of logs.  Each log is a space delimited string of words.

For each log, the first word in each log is an alphanumeric identifier.  Then, either:

Each word after the identifier will consist only of lowercase letters, or;
Each word after the identifier will consist only of digits.
We will call these two varieties of logs letter-logs and digit-logs.  It is guaranteed that each log has at least one 
word after its identifier.

Reorder the logs so that all of the letter-logs come before any digit-log.  The letter-logs are ordered 
lexicographically ignoring identifier, with the identifier used in case of ties.  The digit-logs should be put in their original order.

Return the final order of the logs.

 

Example 1:

Input: logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
Output: ["let1 art can","let3 art zero","let2 own kit dig","dig1 8 1 5 1","dig2 3 6"]
"""

# 819. Most Common Word
"""
Given a paragraph and a list of banned words, return the most frequent word that is not in the list of banned words.  It is guaranteed there is at least one word that isn't banned, and that the answer is unique.

Words in the list of banned words are given in lowercase, and free of punctuation.  Words in the paragraph are not case sensitive.  The answer is in lowercase.

 

Example:

Input: 
paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]
Output: "ball"
"""
class Solution(object):
    def mostCommonWord(self, paragraph, banned):
        """
        :type paragraph: str
        :type banned: List[str]
        :rtype: str
        """
        paragraph = paragraph.lower()
        special_lab = "!,.;?'"
        for s in special_lab:
            if s in paragraph:
                paragraph = paragraph.replace(s, ' ')
        res = paragraph.split()
        p_count = collections.Counter(res)
        p_list = [(ch, c) for ch, c in p_count.items()]
        p_list.sort(key=lambda t: t[1], reverse=True)

        while p_list:
            ch, c = p_list.pop(0)
            if ch in banned:
                continue
            return ch

# 551. Student Attendance Record I
"""
You are given a string representing an attendance record for a student. The record only contains the following three characters:
'A' : Absent.
'L' : Late.
'P' : Present.
A student could be rewarded if his attendance record doesn't contain more than one 'A' (absent) or more than two continuous 'L' (late).

You need to return whether the student could be rewarded according to his attendance record.

Example 1:
Input: "PPALLP"
Output: True
"""


def checkRecord(self, s):
    return s.count('A') <= 1 and 'LLL' not in s
#   return s.count('A')<2 and s.count('LLL')==0
#   return 'LLL' not in s and collections.Counter(s)['A']<=1


# 925. Long Pressed Name
"""
Your friend is typing his name into a keyboard.  Sometimes, when typing a character c, the key might get long pressed, and the character will be typed 1 or more times.

You examine the typed characters of the keyboard.  Return True if it is possible that it was your friends name, with some characters (possibly none) being long pressed.

 

Example 1:

Input: name = "alex", typed = "aaleex"
Output: true
Explanation: 'a' and 'e' in 'alex' were long pressed.
Example 2:

Input: name = "saeed", typed = "ssaaedd"
Output: false
Explanation: 'e' must have been pressed twice, but it wasn't in the typed output.
"""


def isLongPressedName(self, name, typed):
    i = 0
    for j in range(len(typed)):
        if i < len(name) and name[i] == typed[j]:
            i += 1
        elif j == 0 or typed[j] != typed[j - 1]:
            return False
    return i == len(name)


class Solution(object):
    def isLongPressedName(self, name, typed):
        """
        :type name: str
        :type typed: str
        :rtype: bool
        """

        if name == typed:
            return True

        i, j = 0, 0
        stack = []

        while i < len(name) and j < len(typed):
            if name[i] == typed[j]:
                stack.append(typed[j])
                i += 1
                j += 1
            else:
                if len(stack) > 0 and typed[j] == stack[-1]:
                    j += 1
                else:
                    return False

        if i < len(name) and j == len(typed):
            return False
        elif i == len(name) and j < len(typed):
            return all(map(lambda z: z == stack[-1], typed[j:]))
        else:
            return True


class Solution(object):
    def isLongPressedName(self, name, typed):
        """
        :type name: str
        :type typed: str
        :rtype: bool
        """

        slow = 0

        for i in typed:
            if slow < len(name) and name[slow] == i:
                slow += 1

        return slow == len(name)

# 415. Add Strings
"""
Given two non-negative integers num1 and num2 represented as string, return the sum of num1 and num2.

Note:

The length of both num1 and num2 is < 5100.
Both num1 and num2 contains only digits 0-9.
Both num1 and num2 does not contain any leading zero.
You must not use any built-in BigInteger library or convert the inputs to integer directly."""


class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        n1, n2 = 0, 0
        for i in range(len(num1)):
            n1 = n1 * 10 + ord(num1[i]) - 48
        for i in range(len(num2)):
            n2 = n2 * 10 + ord(num2[i]) - 48
        return str(n1 + n2)


class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        num1, num2 = list(num1), list(num2)
        carry, res = 0, []
        while len(num2) > 0 or len(num1) > 0:
            n1 = ord(num1.pop()) - ord('0') if len(num1) > 0 else 0
            n2 = ord(num2.pop()) - ord('0') if len(num2) > 0 else 0

            temp = n1 + n2 + carry
            res.append(temp % 10)
            carry = temp // 10
        if carry: res.append(carry)
        return ''.join([str(i) for i in res])[::-1]


# 345. Reverse Vowels of a String
"""
Write a function that takes a string as input and reverse only the vowels of a string.

Example 1:

Input: "hello"
Output: "holle"
Example 2:

Input: "leetcode"
Output: "leotcede"
Note:
The vowels does not include the letter "y"."""


class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = list(s)
        i = 0
        j = len(s) - 1
        v = set('aeiouAEIOU')
        while True:
            while i < j and s[i] not in v:
                i += 1
            while i < j and s[j] not in v :
                j -= 1
            if i < j:
                s[i], s[j] = s[j], s[i]
                i += 1
                j -= 1
            else:
                break
        return "".join(s)


class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = list(s)
        vowels = set('aeiouAEIOU')
        st, end = 0, len(s) - 1
        while st < end:
            if s[end] not in vowels:
                end -= 1
            elif s[st] not in vowels:
                st += 1
            else:

                s[st], s[end] = s[end], s[st]
                st += 1
                end -= 1
        return "".join(s)


class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        vowels = set(list("aeiouAEIOU"))
        s = list(s)
        ptr_1, ptr_2 = 0, len(s) - 1
        while ptr_1 < ptr_2:
            if s[ptr_1] in vowels and s[ptr_2] in vowels:
                s[ptr_1], s[ptr_2] = s[ptr_2], s[ptr_1]
                ptr_1 += 1
                ptr_2 -= 1
            if s[ptr_1] not in vowels:
                ptr_1 += 1
            if s[ptr_2] not in vowels:
                ptr_2 -= 1
        return ''.join(s)


class Solution(object):
    def reverseVowels(self, s):

        """
    :type s: str
    :rtype: str
    """


        vowel = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}

        s = list(s)

        vpos = []
        vword = []

        for index, char in enumerate(s):
            if char in vowel:
                vpos.append(index)
                vword.append(char)

        for i in vpos:
            s[i] = vword.pop()

        return "".join(s)

# 459. Repeated Substring Pattern
"""
Given a non-empty string check if it can be constructed by taking a substring of it and appending multiple copies of the substring together. You may assume the given string consists of lowercase English letters only and its length will not exceed 10000.

 

Example 1:

Input: "abab"
Output: True
Explanation: It's the substring "ab" twice.
Example 2:

Input: "aba"
Output: False
Example 3:

Input: "abcabcabcabc"
Output: True
Explanation: It's the substring "abc" four times. (And the substring "abcabc" twice.)"""

class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        return s in (2 * s)[1:-1]



def repeatedSubstringPattern(self, s):
    """
    :type s: str
    :rtype: bool
    """
    newstr = s[1:] + s[:-1]
    return newstr.find(s) != -1


class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """

class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        l = len(s)

        for i in xrange(1, l / 2 + 1):
            if l % (i) == 0:
                # print i
                t = l / i
                # print s[:i]
                if s[:i] * t == s: return True
        return False



    for i in range(2, len(s) + 1):
        if len(s) % i != 0: continue
        l = len(s) // i
        sub = s[0:l]
        if ((s[0:l]) * i) == s: return True
    return False


# 67. Add Binary
"""
Given two binary strings, return their sum (also a binary string).

The input strings are both non-empty and contains only characters 1 or 0.

Example 1:

Input: a = "11", b = "1"
Output: "100"
Example 2:

Input: a = "1010", b = "1011"
Output: "10101"
"""


class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """

        def binarytodigit(str):
            k = 0
            c = 0
            str = str[::-1]
            while (k != len(str)):
                c = c + (2 ** k) * int(str[k])
                k = k + 1
            # print(c)
            return c

        def digittobinary(s):
            if s == 0:
                return s
            c = ''
            while (s != 0):
                modc = s
                s = s / 2

                c = c + str(modc % 2)
                print(c)
            return c[::-1]

            print(c[::-1])

        d = binarytodigit(a) + binarytodigit(b)
        print(digittobinary(d))
        return str(digittobinary(d))

class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        return bin(int(a, 2) + int(b, 2))[2:]

class Solution:
    def addBinary(self, a, b):

        if a[-1] == '1' and b[-1] == '1':
            return self.addBinary(self.addBinary(a[0:-1], b[0:-1]), '1') + '0'
        if a[-1] == '0' and b[-1] == '0':
            return self.addBinary(a[0:-1], b[0:-1]) + '0'
        else:
            return self.addBinary(a[0:-1], b[0:-1]) + '1'


# 20. Valid Parentheses
"""
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Note that an empty string is also considered valid.
"""
class Solution:
    def isValid(self, s: str) -> bool:
        bracket_map = {"(": ")", "[": "]",  "{": "}"}
        open_par = set(["(", "[", "{"])
        stack = []
        for i in s:
            if i in open_par:
                stack.append(i)
            elif stack and i == bracket_map[stack[-1]]:
                stack.pop()
            else:
                return False
        return stack == []

class Solution:
    def isValid(self, s: str) -> bool:
        par_dict={
            "(":")",
            "{":"}",
            "[":"]"}
        open_par = set(["(", "[", "{"])
        stack = []
        for i in s:
            if i in open_par:
                stack.append(i)
            elif stack and i==par_dict[stack[-1]]:
                stack.pop()
            else:
                return False
        return stack == []


class Solution:
    def isValid(self, s: str) -> bool:
        start = ["(", "{", "["]
        end = [")", "}", "]"]
        count = 0
        value = []
        if len(s)%2 != 0:
            return False
        for i in range(len(s)):
            if s[i] in start:
                count += start.index(s[i])+1
                value.append(start.index(s[i])+1)
            elif s[i] in end:
                count -= end.index(s[i])+1
                x = value.pop() if value else 0
                if x - end.index(s[i]) - 1 != 0:
                    return False
        if count != 0:
            return False
        else:
            return True


class Solution:
    def isValid(self, s: str) -> bool:
        dic={"]":"[",")":"(","}":"{"}
        stack=[]
        if len(s)==1:
            return False
        for x in s:
            if x in ["[","{","("]:
                stack.append(x)
            else:
                try:
                    query= stack.pop()
                    if dic[x]!=query:
                        return False
                except:
                    return False
        else:
            if stack ==[]:
                return True
            else:
                return False


class Solution:
    # @return a boolean
    def isValid(self, s):
        stack = []
        dict = {"]":"[", "}":"{", ")":"("}
        for char in s:
            if char in dict.values():
                stack.append(char)
            elif char in dict.keys():
                if stack == [] or dict[char] != stack.pop():
                    return False
            else:
                return False
        return stack == []

# 434. Number of Segments in a String
"""Count the number of segments in a string, where a segment is defined to be a contiguous sequence of non-space characters.

Please note that the string does not contain any non-printable characters.

Example:

Input: "Hello, my name is John"
Output: 5
"""

# 125. Valid Palindrome
"""Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

Note: For the purpose of this problem, we define empty string as valid palindrome.

Example 1:

Input: "A man, a plan, a canal: Panama"
Output: true
Example 2:

Input: "race a car"
Output: false
"""
class Solution:
    # @param s, a string
    # @return a boolean
    def isPalindrome(self, s):
        s = "".join([c.lower() for c in s if c.isalnum()])
        return s == s[::-1]


def isPalindrome(self, s):
    l, r = 0, len(s)-1
    while l < r:
        while l < r and not s[l].isalnum():
            l += 1
        while l <r and not s[r].isalnum():
            r -= 1
        if s[l].lower() != s[r].lower():
            return False
        l +=1; r -= 1
    return True


import re
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        string = (re.sub(r"\W+", "", s)).lower()
        return string == string[::-1]


class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not s: return True
        s = (re.sub(r'\W+', '', s)).lower()
        s = s.lower()
        r = s[::-1]

        return r == s




# 680. Valid Palindrome II
"""
Given a non-empty string s, you may delete at most one character. Judge whether you can make it a palindrome.

Example 1:
Input: "aba"
Output: True
Example 2:
Input: "abca"
Output: True
Explanation: You could delete the character 'c'."""

class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # Time: O(n)
        # Space: O(n)
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                one, two = s[left:right], s[left + 1:right + 1]
                return one == one[::-1] or two == two[::-1]
            left, right = left + 1, right - 1
        return True


    def isPalindrome(self, s, start, end, delCount):
        if delCount > 1:
            return False
        while start < end:
            if s[start] != s[end]:
                break
            start += 1
            end -= 1
        if (start == end) or (start == end + 1):
            return True
        return any(
            [self.isPalindrome(s, start + 1, end, delCount + 1), self.isPalindrome(s, start, end - 1, delCount + 1)])

    def validPalindrome(self, s):
        return self.isPalindrome(s, 0, len(s) - 1, 0)


class Solution:
    def validPalindrome(self, s):
        l, r = 0, len(s) - 1
        while l < r:
            if s[l] != s[r]:
                return self.is_valid(s[l:r]) or self.is_valid(s[l + 1:r + 1])
            l += 1;
            r -= 1
        return True

    def is_valid(self, s):
        return s == s[::-1]

#
    def isPalindrome(self, s, start, end, delCount):
        if delCount > 1:
            return False
        while start < end:
            if s[start] != s[end]:
                break
            start += 1
            end -= 1
        if (start == end) or (start == end + 1):
            return True
        return any(
            [self.isPalindrome(s, start + 1, end, delCount + 1), self.isPalindrome(s, start, end - 1, delCount + 1)])

    def validPalindrome(self, s):
        return self.isPalindrome(s, 0, len(s) - 1, 0)


#
def validPalindrome(self, s):
    i, j = 0, len(s) - 1

    while i < j:
        if s[i] == s[j]:
            i += 1
            j -= 1
        else:
            return self.validPalindromeUtil(s, i + 1, j) or self.validPalindromeUtil(s, i, j - 1)
    return True


def validPalindromeUtil(self, s, i, j):
    while i < j:
        if s[i] == s[j]:
            i += 1
            j -= 1
        else:
            return False

    return True

#
class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # Time: O(n)
        # Space: O(n)
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                one, two = s[left:right], s[left + 1:right + 1]
                return one == one[::-1] or two == two[::-1]
            left, right = left + 1, right - 1
        return True


#
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if s == s[::-1]:
            return True
        front_pos = 0
        end_pos = len(s) - 1
        while front_pos < end_pos:
            if s[front_pos] == s[end_pos]:
                front_pos += 1
                end_pos -= 1
            else:
                t1 = s[:front_pos] + s[front_pos + 1:]
                t2 = s[:end_pos] + s[end_pos + 1:]
                return t1 == t1[::-1] or t2 == t2[::-1]
        return True


#
class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        l = 0
        n = len(s)
        r = n - 1
        while l < r:
            if s[l] != s[r]:
                temp1 = s[:l] + s[l + 1:]
                temp2 = s[:r] + s[r + 1:]
                return temp1 == temp1[::-1] or temp2 == temp2[::-1]

            l += 1
            r -= 1
        return True


# 14. Longest Common Prefix
"""Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:

Input: ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
"""
def longestCommonPrefix(self, strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    if not strs:
        return ""
    shortest = min(strs, key=len)
    for i, ch in enumerate(shortest):
        for other in strs:
            if other[i] != ch:
                return shortest[:i]
    return shortest


def longestCommonPrefix(self, strs):
    if not strs:
        return ""

    for i, letter_group in enumerate(zip(*strs)):
        if len(set(letter_group)) > 1:
            return strs[0][:i]
    else:
        return min(strs)



# 28. Implement strStr()
"""
Implement strStr().

Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

Example 1:

Input: haystack = "hello", needle = "ll"
Output: 2
Example 2:

Input: haystack = "aaaaa", needle = "bba"
Output: -1"""

def strStr(self, haystack, needle):
    """
    :type haystack: str
    :type needle: str
    :rtype: int
    """
    for i in range(len(haystack) - len(needle)+1):
        if haystack[i:i+len(needle)] == needle:
            return i
    return -1


class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """

        if needle == '': return 0

        # index of first occurence of needle in haystack or -1
        print(haystack.find(needle))

        return haystack.find(needle)


class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if not needle:
            return 0

        if needle not in haystack:
            return -1

        for i, c in enumerate(haystack):
            if haystack[i:len(needle) + i] == needle:
                return i


# 58. Length of Last Word

# without split
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        if len(s.strip())>0:
            loc=s.strip().rfind(' ')
            if loc == -1:
                return len(s.strip())
            else:
                return len(s.strip()[loc+1:])
        else:
            return 0

class Solution:
	def lengthOfLastWord(self, s: str) -> int:
		approach1
		return len(s.strip().split(" ")[-1])

		approach2
		count = 0
		for i in range(len(s)-1,-1,-1):
			if not s[i] == ' ':
				count += 1
			elif s[i] == ' ' and count > 0:
				return count
		return count


	   #approach 3
		if len(s) == 0 :
			return 0
		words = s.split(" ")
		count = 0
		for i in words:
			if not i == '':
				count = len(i)
		return count

	  #approach 4
		l =s.split()
		print(len(l))
		if len(l)==0:
			return 0
		return len(l[-1])

# 686. Repeated String Match
"""
Given two strings A and B, find the minimum number of times A has to be repeated such that B is a substring of it. If no such solution, return -1.

For example, with A = "abcd" and B = "cdabcdab".

Return 3, because by repeating A three times (“abcdabcdabcd”), B is a substring of it; and B is not a substring of A repeated two times ("abcdabcd").

Note:
The length of A and B will be between 1 and 10000."""

def repeatedStringMatch(A, B):
	count=1
	cur =  A
	while B not in cur:
		cur += A
		count += 1
		if count > len(B)//len(A) + 2:
			print (len(B)//len(A))
			return -1

class Solution(object):
    def repeatedStringMatch(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: int
        """
        l1 = len(A)
        l2 = len(B)
        k = l2//l1
        for i in range(k,k+3):
            temp = A*i
            if B in temp:
                return i

        return -1





