def stack_generator(stack):
    while stack:
        yield stack.pop()

# Example usage
stack = [1, 2, 3, 4, 5]

for value in stack_generator(stack):
    print(value)
