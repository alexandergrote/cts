from collections import defaultdict

def prefixspan(prefix, projected_db, min_support, max_length, classes):
    """
    Recursive function that generates sequential patterns using the PrefixSpan algorithm.
    
    :param prefix: Current prefix (a list of items).
    :param projected_db: Projected database containing only sequences that start with the prefix.
    :param min_support: Minimum support threshold for patterns.
    :return: A list of frequent sequential patterns.
    """

    assert len(projected_db) == len(classes)

    patterns = []
    freq_items = defaultdict(int)
    freq_items_pos = defaultdict(int)
    freq_items_neg = defaultdict(int)

    # can't i just count the class values here as well?
    # maybe pass a mask of the class values as an argument

    # Count support for each item in the projected database
    for sequence, class_value in zip(projected_db, classes):

        used = set()

        for item in sequence:

            if item in used:
                continue

            freq_items[item] += 1

            if class_value == 0:
                freq_items_neg[item] += 1

            if class_value == 1:
                freq_items_pos[item] += 1

            used.add(item)

    #print("x"*10)
    #print(freq_items)

    # since it is binary, we can sum up the class values to obtain the frequency of the sequence

    # Check each frequent item and recursively grow the pattern
    for item, count in freq_items.items():

        if count >= min_support:

            new_prefix = prefix + [item]
            patterns.append((new_prefix, count, freq_items_neg.get(item, 0), freq_items_pos.get(item, 0)))

            # Project the database for the new prefix
            new_projected_db = []
            new_classes = []

            for sequence, class_value in zip(projected_db, classes):

                try:
                    index = sequence.index(item)
                    new_projected_db.append(sequence[index+1:])
                    new_classes.append(class_value)
                except ValueError:
                    continue
            
            #print("-"*10)
            #print(new_projected_db)

            # if the new projected db does not have enough entries
            # that could satisfy the min support threshold
            # continue with next iteration
            if len(new_projected_db) < min_support:
                continue

            # Recursive call
            patterns.extend(prefixspan(new_prefix, new_projected_db, min_support, max_length, new_classes))

    return patterns


def prefixspan_iterative(database, min_support, max_length, classes):

    assert len(database) == len(classes)

    # prepare data for while loop
    # more memory efficient than recursion - prevents stack overflow
    stack = [([], database, classes)]
    patterns = []

    while stack:

        prefix, projected_db, projected_classes = stack.pop()

        freq_items = defaultdict(int)
        freq_items_pos = defaultdict(int)
        freq_items_neg = defaultdict(int)

        assert len(projected_db) == len(projected_classes)

        for sequence, class_value in zip(projected_db, projected_classes):

            used = set()
            for item in sequence:
                if item in used:
                    continue

                freq_items[item] += 1

                if class_value == 0:
                    freq_items_neg[item] += 1

                if class_value == 1:
                    freq_items_pos[item] += 1

                used.add(item)

        for item, count in freq_items.items():

            if count < min_support:
                continue

            if len(prefix) + 1 > max_length:
                continue

            new_prefix = prefix + [item]
            
            patterns.append((new_prefix, count, freq_items_neg.get(item, 0), freq_items_pos.get(item, 0)))

            new_projected_db = []
            new_classes = []

            for sequence, class_value in zip(projected_db, projected_classes):

                try:

                    index = sequence.index(item)
                    new_projected_db.append(sequence[index + 1:])
                    new_classes.append(class_value)

                except ValueError:
                    continue

            stack.append((new_prefix, new_projected_db, new_classes))

    return patterns


def yield_iterative_prefixspan(database, min_support, classes):

    stack = [([], database, classes)]
    patterns = []

    def stack_generator(stack):
        while stack:

            prefix, projected_db, projected_classes = stack.pop()

            freq_items = defaultdict(int)
            freq_items_pos = defaultdict(int)
            freq_items_neg = defaultdict(int)

            assert len(projected_db) == len(projected_classes)

            for sequence, class_value in zip(projected_db, projected_classes):

                used = set()
                
                for item in sequence:
                    if item in used:
                        continue

                    freq_items[item] += 1

                    if class_value == 0:
                        freq_items_neg[item] += 1

                    if class_value == 1:
                        freq_items_pos[item] += 1

                    used.add(item)

            for item, count in freq_items.items():

                if count >= min_support:

                    new_prefix = prefix + [item]
                    yield (new_prefix, count, freq_items_neg.get(item, 0), freq_items_pos.get(item, 0))

                    new_projected_db = []
                    new_classes = []

                    for sequence, class_value in zip(projected_db, projected_classes):

                        try:

                            index = sequence.index(item)
                            new_projected_db.append(sequence[index + 1:])
                            new_classes.append(class_value)

                        except ValueError:
                            continue

                    stack.append((new_prefix, new_projected_db, new_classes))

    for value in stack_generator(stack):
        patterns.append(value)


    return patterns


def main():

    # Example dataset: a list of sequences
    database = [
        ['A', 'B', 'C', 'D'],
        ['A', 'B', 'C', 'D'],
        ['A', 'C', 'B', 'E'],
        ['A', 'B', 'C', 'E'],
        ['B', 'C', 'E'],
        ['A', 'B', 'D', 'E']
    ]

    classes = [
        0, 1, 0, 1, 1, 1
    ]

    min_support = 2  # Minimum support threshold
    patterns = prefixspan([], database, min_support, 3, classes)
    patterns = prefixspan_iterative(database=database, min_support=min_support, max_length=3, classes=classes)
    #patterns = yield_iterative_prefixspan(database=database, min_support=min_support, classes=classes)

    # Display the patterns
    #for pattern, support, support_neg, support_pos in patterns:
    for pattern, support, support_neg, support_pos in patterns:
        print(f"Pattern: {pattern}, Support: {support}, Support Neg: {support_neg}, Support Pos: {support_pos}")


if __name__ == "__main__":
    main()
