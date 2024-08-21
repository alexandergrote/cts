def debug_iter(iterator):
    while True:
         value = next(iterator)
         print('Iterator yielded', repr(value))
         yield value

def is_subsequence(subseq, seq):
    it = iter(seq)
    print(it)
    return all(item in it for item in subseq)

# Example usage:
sequence = ['x', 'a', 'c', 'b', 'a']
subsequence = ['a', 'b']

print(is_subsequence(subsequence, sequence))  # Output: True
print(is_subsequence(['b', 'd'], sequence))  # Output: True
