class NumArray(object):
    def __init__(self, nums):
        """
        initialize your data structure here.
        :type nums: List[int]
        """
        self.nums = nums
        sums = {}
        l = len(nums)
        for i in range(l):
            for j in range(i, l):
                sums[(i,j)] = sum([n for n in nums[i: j+1]])
        self.sums = sums
        

    def sumRange(self, i, j):
        """
        sum of elements nums[i..j], inclusive.
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.sums.get((i,j))