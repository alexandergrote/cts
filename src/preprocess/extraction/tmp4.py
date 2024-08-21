from collections import defaultdict

def prefixspan_iterative(database, min_support, classes):
    assert len(database) == len(classes)

    # Prepare the initial stack
    stack = [([], database, classes)]

    while stack:
        prefix, projected_db, projected_classes = stack.pop()

        freq_items = defaultdict(int)
        freq_items_pos = defaultdict(int)
        freq_items_neg = defaultdict(int)

        assert len(projected_db) == len(projected_classes)

        # Count the frequency of items
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

        # Generate new patterns and yield them
        for item, count in freq_items.items():
            if count >= min_support:
                new_prefix = prefix + [item]
                yield (new_prefix, count, freq_items_neg.get(item, 0), freq_items_pos.get(item, 0))

                new_projected_db = []
                new_classes = []

                # Construct the new projected database and classes
                for sequence, class_value in zip(projected_db, projected_classes):
                    try:
                        index = sequence.index(item)
                        new_projected_db.append(sequence[index + 1:])
                        new_classes.append(class_value)
                    except ValueError:
                        continue

                # Instead of appending directly to the stack, we yield and continue processing
                stack.append((new_prefix, new_projected_db, new_classes))

        # Yield and process the next stack item using the outer loop
        while stack:
            yield from prefixspan_iterative(*stack.pop())

# Example usage:
# for pattern in prefixspan_iterative(database, min_support, classes):
#     print(pattern)

def main():

    # Example dataset: a list of sequences
    database = [
        ['A', 'B', 'C', 'D'],
        ['A', 'B', 'C', 'D'],
        ['A', 'C', 'B', 'E'],
        ['A', 'B', 'C', 'B'],
        ['B', 'C', 'E'],
        ['A', 'B', 'D', 'E']
    ]

    classes = [
        0, 1, 0, 1, 1, 1
    ]

    min_support = 2  # Minimum support threshold
    patterns = prefixspan_iterative(database=database, min_support=min_support, classes=classes)

    # Display the patterns
    for pattern, support, support_neg, support_pos in patterns:
        print(f"Pattern: {pattern}, Support: {support}, Support Neg: {support_neg}, Support Pos: {support_pos}")


if __name__ == "__main__":
    main()
