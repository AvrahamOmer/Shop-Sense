"""Module priting hello world."""
import os
print("hello-world")

# create function to print the current path
def print_path():
    """Function printing path."""
    print(os.getcwd())

print_path()
