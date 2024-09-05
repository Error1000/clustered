import argparse
from pathlib import Path
import random
from subprocess import Popen, PIPE, STDOUT 
import multiprocessing as mp
import tqdm
import threading
import datetime

parser = argparse.ArgumentParser("fuzzer")
parser.add_argument("program_1", help="The path to program 1.", type=str)
parser.add_argument("program_2", help="The path to program 2.", type=str)
parser.add_argument("--n_iter", help="Number of inputs to generate and try.", type=int, default=100, required=False)
parser.add_argument("--verbose", "-v", help="Print entire output on mismatch.", action='store_true', required=False)
args = parser.parse_args()

def gen_input():
    # NOTE: This needs to be changed depending on the program you are fuzzing!
    n = random.randrange(1, 500_000)
    ginput = ""
    ginput += str(n) + '\n'
    return ginput


print_lock = threading.Lock()

from decimal import *

def are_results_equivalent(rez1, rez2):
    rez1_processed = [ line for line in rez1.decode('utf-8').split('\n') if line.strip() != '' and line.strip()[0].isnumeric() ]
    rez2_processed = [ line for line in rez2.decode('utf-8').split('\n') if line.strip() != '' and line.strip()[0].isnumeric() ]
    mat_a = [[float(e) for e in line.split(' ') if e.strip() != ''] for line in rez1_processed]
    mat_b = [[float(e) for e in line.split(' ') if e.strip() != ''] for line in rez2_processed]
    nrows = len(mat_a)
    assert nrows == len(mat_b)
    ncols = len(mat_a[0])

    for i in range(nrows):
        for j in range(ncols):
            assert len(mat_a[i]) == ncols
            assert len(mat_b[i]) == ncols
            diff = abs(mat_a[i][j] - mat_b[i][j])
            if diff > 0.01:
                return False
    return True
# Returns wether or not programs being fuzzed have the same output (a.k.a behaviour)
def try_input(nid):
    gin = gen_input()

    pipe1 = Popen(args.program_1, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    rez1 = pipe1.communicate(input=gin.encode())
    if pipe1.returncode != 0:
        print(f"Program {args.program_1.split('/')[-1]} didn't return exit code 0, it returned {pipe1.returncode} for input:\n {gin}")
        return False
    if rez1[1] != None:
        print(f"Program {args.program_1.split('/')[-1]} printed to stderr:")
        print(rez1[1])
        return False

    pipe2 = Popen(args.program_2, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    rez2 = pipe2.communicate(input=gin.encode())
        

    if pipe2.returncode != 0:
        print(f"Program {args.program_2.split('/')[-1]} didn't return exit code 0, it returned {pipe2.returncode} for input:\n {gin}")
        return False
    if rez2[1] != None:
        print(f"Program {args.program_2.split('/')[-1]} printed to stderr:")
        print(rez2[1])
        return False

    if not are_results_equivalent(rez1[0], rez2[0]):
        with print_lock:
            print(f"Programs executed succesfully(exitcode = 0) but disagree for input: \n {gin}")
            if args.verbose:
                print(f"Program {args.program_1.split('/')[-1]} said: ", rez1)
                print(f"Program {args.program_2.split('/')[-1]} said: ", rez2)
        return False
    else:
        return True

n_iter = args.n_iter
found_bad_input = False

import math
if __name__ == '__main__':
        with mp.Pool(math.ceil(mp.cpu_count()/2)) as pool:
                for result in tqdm.tqdm(pool.imap_unordered(try_input, range(n_iter)), total=n_iter):
                    if result == False:
                        pool.terminate()
                        found_bad_input = True
                        break
        if not found_bad_input: print("Done fuzzing! No bad inputs found, yay :)")