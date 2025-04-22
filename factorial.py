"""This module provides a function to calculate the factorial of a non-negative integer."""

def factorial(number):
    """Calculate factorial of a non-negative integer."""
    if number < 0:
        raise ValueError("Negative numbers are not allowed.")
    if number in (0, 1):
        return 1
    return number * factorial(number - 1)

if __name__ == "__main__":
    NUMBER = 5
    print(f"Factorial of {NUMBER} is {factorial(NUMBER)}")
    