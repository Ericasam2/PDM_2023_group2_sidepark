import math

import sys


def equation(v, a, t):
    u = v - a*t
    return u

def main():
    # Check if at least one command-line argument is provided
    if len(sys.argv) > 1:
        # Access individual command-line arguments
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        arg3 = sys.argv[3]
    else:
        print("No command-line arguments provided.")
    
    v = int(arg1)
    a = int(arg2)
    t = int(arg3)
    u = equation(v,a,t)
    print(u)
    
if __name__ == '__main__':
    main()
    