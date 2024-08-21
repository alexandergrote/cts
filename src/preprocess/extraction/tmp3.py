class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_sequence = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, sequence):
        current_node = self.root
        for item in sequence:
            if item not in current_node.children:
                current_node.children[item] = TrieNode()
            current_node = current_node.children[item]
        current_node.is_end_of_sequence = True

    def get_suffixes(self, prefix):
        current_node = self.root
        for item in prefix:
            if item not in current_node.children:
                return []
            current_node = current_node.children[item]
        return self._collect_suffixes(current_node, prefix[len(prefix):])

    def _collect_suffixes(self, node, suffix):
        suffixes = []
        if node.is_end_of_sequence:
            suffixes.append(suffix)
        for item, next_node in node.children.items():
            suffixes.extend(self._collect_suffixes(next_node, suffix + [item]))
        return suffixes

class PrefixSpan:
    def __init__(self, min_support):
        self.min_support = min_support
        self.frequent_patterns = []

    def fit(self, sequences):
        self.trie = Trie()
        for seq in sequences:
            self.trie.insert(seq)
        self.prefix_span([], self.trie.root)

    def prefix_span(self, prefix, trie_node):
        frequent_items = self.find_frequent_items(trie_node)
        for item in frequent_items:
            new_prefix = prefix + [item]
            self.frequent_patterns.append(new_prefix)
            new_trie_node = self.project_trie(trie_node, item)
            if new_trie_node:
                self.prefix_span(new_prefix, new_trie_node)

    def find_frequent_items(self, trie_node):
        item_counts = {}
        self._count_items(trie_node, item_counts)
        return [item for item, count in item_counts.items() if count >= self.min_support]

    def _count_items(self, node, item_counts):
        if node.is_end_of_sequence:
            return
        for item, next_node in node.children.items():
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1
            self._count_items(next_node, item_counts)

    def project_trie(self, node, item):
        if item not in node.children:
            return None
        return node.children[item]

    def get_frequent_patterns(self):
        return self.frequent_patterns

# Example Usage:
sequences = [
    ['a', 'b', 'c'],
    ['a', 'b', 'd'],
    ['a', 'e'],
    ['b', 'c', 'd'],
]

min_support = 2
prefix_span = PrefixSpan(min_support)
prefix_span.fit(sequences)
frequent_patterns = prefix_span.get_frequent_patterns()

print("Frequent Patterns:", frequent_patterns)
