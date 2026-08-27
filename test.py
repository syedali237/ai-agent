import os

from functions import get_files_info

def main():
    working_directory = os.getcwd()
    files_info = get_files_info(working_directory)
    print(files_info)

if __name__ == "__main__":
    main()