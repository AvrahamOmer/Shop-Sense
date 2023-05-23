"""Module priting hello world."""
import os
print("hello-world")

# create function to print the current path
def print_path():
    """Function printing path."""
    print(os.getcwd())

#create function to print the username
def print_username():
    """Function printing username."""
    print(os.getlogin())

print_username()
print_path()
