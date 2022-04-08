import argparse
import runpy

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

for i in range(1, 7):
    runpy.run_module(f"src.figure{i}")

for i in range(1, 9):
    runpy.run_module(f"src.suppfig{i}")
