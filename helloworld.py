"""Module priting hello world."""
import os
print("hello-world")

# create function to print the current path
def print_path():
    #Print the current path.
    print(os.getcwd())

print_path()
