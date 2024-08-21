from collections import defaultdict

def prefixspan(prefix, projected_db, min_support):
    """
    Recursive function that generates sequential patterns using the PrefixSpan algorithm.
    
    :param prefix: Current prefix (a list of items).
    :param projected_db: Projected database containing only sequences that start with the prefix.
    :param min_support: Minimum support threshold for patterns.
    :return: A list of frequent sequential patterns.
    """
    patterns = []
    freq_items = defaultdict(int)

    # can't i just count the class values here as well?
    # maybe pass a mask of the class values as an argument

    # Count support for each item in the projected database
    for sequence in projected_db:
        used = set()
        for item in sequence:
            if item not in used:
                freq_items[item] += 1
                used.add(item)

    print("x"*10)
    print(freq_items)

    # since it is binary, we can sum up the class values to obtain the frequency of the sequence

    # Check each frequent item and recursively grow the pattern
    for item, count in freq_items.items():
        if count >= min_support:
            new_prefix = prefix + [item]
            patterns.append((new_prefix, count))

            # Project the database for the new prefix
            new_projected_db = []
            for sequence in projected_db:
                try:
                    index = sequence.index(item)
                    new_projected_db.append(sequence[index+1:])
                except ValueError:
                    continue
            
            print("-"*10)
            print(new_projected_db)

            # Recursive call
            patterns.extend(prefixspan(new_prefix, new_projected_db, min_support))

    return patterns

def main():

    # Example dataset: a list of sequences
    database = [
        ['A', 'B', 'C', 'D'],
        ['A', 'C', 'B', 'E'],
        ['A', 'B', 'C', 'B'],
        ['B', 'C', 'E'],
        ['A', 'B', 'D', 'E']
    ]

    min_support = 2  # Minimum support threshold
    patterns = prefixspan([], database, min_support)

    # Display the patterns
    for pattern, support in patterns:
        print(f"Pattern: {pattern}, Support: {support}")

if __name__ == "__main__":
    main()
