#!/bin/python3
import os
import re
import argparse

def replace_case_sensitive(text, old, new):
    """Replace 'old' with 'new' while maintaining the case in 'text'."""
    
    def match_case(m):
        match = m.group()
        if match.islower():
            return new.lower()
        else:
            return new  # Default case
    
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    return pattern.sub(match_case, text)


def rename_files_and_dirs(root_dir, old, new):
    """Recursively rename files and directories replacing 'old' with 'new' while preserving case."""
    
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Rename files
        for filename in filenames:
            new_filename = replace_case_sensitive(filename, old, new)
            if filename != new_filename:
                os.rename(os.path.join(dirpath, filename), os.path.join(dirpath, new_filename))
        
        # Rename directories
        for dirname in dirnames:
            new_dirname = replace_case_sensitive(dirname, old, new)
            if dirname != new_dirname:
                os.rename(os.path.join(dirpath, dirname), os.path.join(dirpath, new_dirname))

def replace_in_file(file_path, old, new):
    """Replace occurrences of 'old' with 'new' inside a file while preserving case."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_content = replace_case_sensitive(content, old, new)
        
        if content != new_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")

def process_directory(root_dir, old, new):
    """Process all files and directories for renaming and content replacement."""
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            replace_in_file(file_path, old, new)
    
    rename_files_and_dirs(root_dir, old, new)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively rename files, directories, and replace inside files while preserving case.")
    parser.add_argument("root_dir", help="Root directory to start renaming")
    parser.add_argument("new", help="New string to replace with")
    parser.add_argument("old", default="standalone", type=str, help="Old string to replace")
    args = parser.parse_args()
    args.old = args.old.lower()

    process_directory(args.root_dir, args.old, args.new)
    print("Renaming and content replacement complete!")
