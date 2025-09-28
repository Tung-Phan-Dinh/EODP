"""
COMP20008 Elements of Data Processing
2025 Semester 2
Assignment 1

Solution: main file

DO NOT CHANGE THIS FILE!
"""

import os
import sys

from datetime import datetime  
print(datetime.now())   

def verify_task3_1():
    try:
        from task3_1 import task3_1
    except ImportError:
        print("Task 3.1's function not found.")
        return

    print("=" * 80)
    print("Executing Task 3.1...\n")
    task3_1()

    print("Checking Task 3.1's output...\n")
    for expected_file in ["task3_1.png"]:
        if os.path.isfile(expected_file):
            print(f"\tTask 3.1's {expected_file} output found.\n")
            if os.path.getsize(expected_file) == 0:
                print(f"\t❗ Task 3.1's {expected_file} output has size zero - please verify it uploaded correctly.\n")
        else:
            print(f"\t❗ Task 3.1's {expected_file} output NOT found. Please check your code.\n")

    print("Finished Task 3.1")
    print("=" * 80)


def verify_task3_2():

    try:
        from task3_2 import task3_2
    except ImportError:
        print("Task 3.2's function not found.")
        return

    print("=" * 80)
    print("Executing Task 3.2...\n")
    task3_2()

    print("Checking Task 3.2's output...\n")
    for expected_file in ["task3_2.png"]:
        if os.path.isfile(expected_file):
            print(f"\tTask 3.2's {expected_file} output found.\n")
            if os.path.getsize(expected_file) == 0:
                print(f"\t❗ Task 3.2's {expected_file} output has size zero - please verify it uploaded correctly.\n")
        else:
            print(f"\t❗ Task 3.2's {expected_file} output NOT found. Please check your code.\n")

    print("Finished Task 3.2")
    print("=" * 80)


def main():
    args = sys.argv
    assert len(args) >= 1, "Please provide a task."
    task = args[1]
    valid_tasks = ["all"] + ["task3_" + str(i) for i in range(1, 4)]
    assert task in valid_tasks, \
        f"Invalid task \"{task}\", options are: {valid_tasks}."
    if task == "task3_1":
         verify_task3_1()
    elif task == "task3_2":
         verify_task3_2()
    elif task == "all":
         verify_task3_1()
         verify_task3_2()

if __name__ == "__main__":
    main()

