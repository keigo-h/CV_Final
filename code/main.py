from line_translation import run_translation
import sys
import cv2

"""
Main function
"""

def main():
    arguments = sys.argv[1:]
    if len(arguments) != 5:
        print("<file path> <file type> <model file path> <destination language> <verbose>")
        sys.exit()
    path = arguments[0]
    if cv2.imread(path) is None:
        print("File path is invalid")
        sys.exit() 
    file_type = int(arguments[1])
    if file_type != 0 and file_type != 1 and file_type != 2:
        print("Invalid File path type")
        print("File types: 0 multi line, 1 single line, 2 single word")
        sys.exit()
    model_file_path = arguments[2]
    dest_language = arguments[3]
    verbose = int(arguments[4])
    run_translation(path, file_type, model_file_path, dest_language, verbose)

main()