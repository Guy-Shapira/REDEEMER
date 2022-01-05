import os
import sys
from pathlib import Path
import fileinput
from glob import glob



def main():
    path = Path(__file__).parent.absolute()
    # file_name = "CEP.py"
    dirs = ["adaptive", "base", "evaluation", "misc", "parallel", "plan", "plugin", "stream", "test", "transformation", "tree", "condition"]
    # dirs = ["condition"]
    results = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.py'))]

    for file_name in results:
        curr_path = os.path.join(path, file_name)

        with open(curr_path, 'r') as file :
            filedata = file.read()

        # a = filedata.find("from OPEN_CEP.misc import")
        # Replace the target string
        for dir in dirs:
            filedata = filedata.replace(f"from {dir}", f"from OPEN_CEP.{dir}")

        # Write the file out again
        with open(curr_path, 'w') as file:
          file.write(filedata)

if __name__ == "__main__":
    main()
