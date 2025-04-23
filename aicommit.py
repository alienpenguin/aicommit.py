#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
import sys
import datetime # Added for timestamping the saved message file
from pathlib import Path # Added for handling home directory path

try:
    # Attempt to import the OpenAI library
    from openai import OpenAI, OpenAIError
except ImportError:
    # Handle the case where the library is not installed
    print("Error: The 'openai' library is not installed.")
    print("Please run: pip install openai")
    sys.exit(1)

# --- Configuration ---
# OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo")
OPENAI_MODEL = "gpt-4o-mini"
# Prompt template for OpenAI
OPENAI_PROMPT_TEMPLATE = """
You are a skilled developer specializing in writing excellent commit messages following the Conventional Commits standard (https://www.conventionalcommits.org/).

Analyze the following git patch and generate an appropriate commit message.

The format MUST be:
<type>[optional scope]: <description>

[optional body explaining the 'what' and 'why' vs. 'how']

[optional footer(s) for BREAKING CHANGE or referencing issues (this MUST be present ONLY if the branch name contains a numer like ISSUE-NNN), e.g., Closes #123]

Common types include: feat, fix, build, chore, ci, docs, style, refactor, perf, test.

Here is the git patch:
```diff
{}
```

Generate ONLY the commit message itself, starting directly with the type. Do not include any introductory phrases like "Here is the commit message:".
"""
# --- End Configuration ---

def run_git_command(command):
    """Executes a Git command and returns the output or None on error."""
    try:
        # Use utf-8 to handle special characters in filenames and messages
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        return result.stdout.strip()
    except FileNotFoundError:
        print(f"Error: The command '{command[0]}' was not found.")
        print("Ensure Git is installed and available in the system PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing '{' '.join(command)}':")
        # Print stderr to provide more context on the Git error
        print(e.stderr, file=sys.stderr)
        return None

def is_git_repository():
    """Checks if the current directory is a Git repository."""
    # Use 'git rev-parse --is-inside-work-tree' which is a standard way
    # to check if we are inside a Git work tree.
    # It returns 'true' if it is, otherwise fails or returns nothing.
    result = run_git_command(['git', 'rev-parse', '--is-inside-work-tree'])
    return result == 'true'

def get_staged_diff():
    """Retrieves changes in the staging area (git diff --cached)."""
    return run_git_command(['git', 'diff', '--cached'])

def get_openai_api_key():
    """Retrieves the OpenAI API key from environment variable or fallback file."""
    # 1. Try environment variable first
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        # print("DEBUG: Found API key in environment variable.") # Optional debug message
        return api_key

    # 2. If not found, try the fallback file
    fallback_path = Path.home() / ".nr" / "aicommit" / "openai_api_key"
    # print(f"DEBUG: Checking fallback path: {fallback_path}") # Optional debug message
    if fallback_path.is_file():
        try:
            api_key = fallback_path.read_text(encoding='utf-8').strip()
            if api_key:
                 # print("DEBUG: Found API key in fallback file.") # Optional debug message
                 return api_key
            else:
                 print(f"Warning: Fallback API key file found but is empty: {fallback_path}")
        except IOError as e:
            print(f"Warning: Could not read fallback API key file at {fallback_path}: {e}")
        except Exception as e:
             print(f"Warning: An unexpected error occurred reading fallback API key file: {e}")

    # 3. If key is not found in either location
    print("Error: OpenAI API key not found.")
    print("Please set the OPENAI_API_KEY environment variable or")
    print(f"place your key in the file: {fallback_path}")
    return None


def get_openai_commit_message(patch):
    """Calls the OpenAI API to generate the commit message."""
    api_key = get_openai_api_key() # Use the new function to get the key
    if not api_key:
        # Error message is already printed by get_openai_api_key()
        return None

    try:
        client = OpenAI(api_key=api_key)
        full_prompt = OPENAI_PROMPT_TEMPLATE.format(patch)

        print(f"Requesting commit message from OpenAI ({OPENAI_MODEL})...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                # Provide context to the model about its task
                {"role": "system", "content": "You generate Conventional Commit messages based on git diffs."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.5,  # A medium value for a good balance between creativity and consistency
            max_tokens=200    # Limit the length of the generated message
        )

        # Extract the message content from the response
        if response.choices and response.choices[0].message:
            message = response.choices[0].message.content.strip()
             # Sometimes models add ``` at the start or end
            if message.startswith("```") and message.endswith("```"):
                message = message[3:-3].strip()
            return message
        else:
            print("Error: Unexpected response from the OpenAI API.")
            print(response)
            return None

    except OpenAIError as e:
        print(f"Error calling the OpenAI API: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during OpenAI interaction: {e}")
        return None

def save_message_to_file(message):
    """Saves the commit message to a timestamped file."""
    try:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S") # Changed format slightly for better readability
        filename = f"aicommit_msg_{timestamp}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(message)
        print(f"Commit message saved to: {filename}")
        return True
    except IOError as e:
        print(f"Error saving message to file {filename}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while saving the file: {e}")
        return False

def main():
    """Main function of the script."""
    print("Checking Git repository...")
    if not is_git_repository():
        print("Error: This directory does not appear to be a Git repository.")
        sys.exit(1)
    print("OK.")

    print("Retrieving staged changes (git diff --cached)...")
    patch = get_staged_diff()

    if patch is None:
        # The specific error was already printed by run_git_command
        sys.exit(1)

    if not patch:
        print("No changes staged for commit (did you forget to 'git add'?).")
        print("Nothing to commit.")
        sys.exit(0)
    print("OK. Found staged changes.")

    # This function now handles getting the API key from env var or file
    commit_message = get_openai_commit_message(patch)

    if not commit_message:
        print("Could not generate commit message. Aborting.")
        sys.exit(1)

    print("\n" + "="*20 + " Suggested Commit Message " + "="*20)
    print(commit_message)
    print("=" * (42 + len(" Suggested Commit Message ")) + "\n") # Matching closing line

    # Ask the user for confirmation with the new 'save' option
    while True:
        # Updated prompt to include the 's' option
        confirm = input("Use this message? (y/n/s[ave]): ").lower().strip()
        if confirm == 'y':
            # Proceed with commit
            break
        elif confirm == 'n':
            # Cancel commit
            print("Commit cancelled by user.")
            sys.exit(0)
        elif confirm == 's':
            # Save the message to a file and exit
            if save_message_to_file(commit_message):
                print("Exiting without committing.")
            else:
                # Inform user if saving failed, but still exit as requested
                print("Exiting despite save error.")
            sys.exit(0) # Exit after saving or attempting to save
        else:
            # Handle invalid input
            print("Invalid input. Please enter 'y' (yes), 'n' (no), or 's' (save).")

    # Execute the commit (only reached if user entered 'y')
    print("\nExecuting git commit...")
    # Use '-m' which allows passing the message directly.
    # Git handles multiline messages passed with -m correctly.
    commit_result = run_git_command(['git', 'commit', '-m', commit_message])

    if commit_result is not None:
        print("\nCommit successful!")
        print("-" * 10 + " Git Output " + "-" * 10)
        print(commit_result) # Show the standard output of the git commit command
        print("-" * (20 + len(" Git Output ")))
    else:
        print("\nError during commit execution.")
        print("The message might have been too complex or contained problematic characters.")
        print("You can try copying the suggested message and running 'git commit' manually.")
        sys.exit(1)

if __name__ == "__main__":
    main()
