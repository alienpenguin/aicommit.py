#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
import sys
import datetime
import argparse # Added for command-line arguments
from pathlib import Path

# --- Import AI Libraries ---
# Attempt to import necessary libraries. Handle missing ones gracefully.
try:
    from openai import OpenAI, OpenAIError
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: The 'openai' library is not installed. OpenAI engine will not be available.")
    print("To install: pip install openai")
    OPENAI_AVAILABLE = False

try:
    # Assuming google-generativeai for Gemini. You might need a different library.
    # To install: pip install google-generativeai
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Warning: The 'google-generativeai' library is not installed. Gemini engine will not be available.")
    print("To install: pip install google-generativeai")
    GEMINI_AVAILABLE = False

try:
    # Assuming anthropic for Claude. You might need a different library or direct HTTP.
    # To install: pip install anthropic
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    print("Warning: The 'anthropic' library is not installed. Claude engine will not be available.")
    print("To install: pip install anthropic")
    CLAUDE_AVAILABLE = False

# --- Configuration ---
# Default AI engine
DEFAULT_ENGINE = "openai"

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

# --- Placeholder Configurations for Other Engines ---
# You will need to configure models and prompt templates for Gemini and Claude
GEMINI_MODEL = "gemini-pro" # Example model name
GEMINI_PROMPT_TEMPLATE = """
Generate a Conventional Commit message based on the following git patch.
Follow the format: <type>[optional scope]: <description>\n\n[optional body]\n\n[optional footer(s)]
Common types: feat, fix, build, chore, ci, docs, style, refactor, perf, test.
Include a footer like 'Closes #NNN' if the branch name contains 'ISSUE-NNN'.

Git patch:
```diff
{}
```
Output only the commit message.
"""

CLAUDE_MODEL = "claude-3-opus-20240229" # Example model name
CLAUDE_PROMPT_TEMPLATE = """
Please provide a Conventional Commit message for the following git patch.
The message should adhere to the format: <type>[optional scope]: <description>\n\n[optional body]\n\n[optional footer(s)].
Standard types include: feat, fix, build, chore, ci, docs, style, refactor, perf, test.
If the branch name includes 'ISSUE-NNN', add a footer like 'Closes #NNN'.

Here is the git patch:
```diff
{}
```
Generate only the commit message content.
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
    result = run_git_command(['git', 'rev-parse', '--is-inside-work-tree'])
    return result == 'true'

def get_staged_diff():
    """Retrieves changes in the staging area (git diff --cached)."""
    return run_git_command(['git', 'diff', '--cached'])

def get_api_key(engine):
    """Retrieves the API key for the specified engine."""
    env_var = f"{engine.upper()}_API_KEY"
    api_key = os.getenv(env_var)
    if api_key:
        # print(f"DEBUG: Found API key for {engine} in environment variable.") # Optional debug message
        return api_key

    # Fallback file path
    fallback_path = Path.home() / ".nr" / "aicommit" / f"{engine}_api_key"
    # print(f"DEBUG: Checking fallback path for {engine}: {fallback_path}") # Optional debug message
    if fallback_path.is_file():
        try:
            api_key = fallback_path.read_text(encoding='utf-8').strip()
            if api_key:
                 # print(f"DEBUG: Found API key for {engine} in fallback file.") # Optional debug message
                 return api_key
            else:
                 print(f"Warning: Fallback API key file for {engine} found but is empty: {fallback_path}")
        except IOError as e:
            print(f"Warning: Could not read fallback API key file for {engine} at {fallback_path}: {e}")
        except Exception as e:
             print(f"Warning: An unexpected error occurred reading fallback API key file for {engine}: {e}")

    print(f"Error: {engine.capitalize()} API key not found.")
    print(f"Please set the {env_var} environment variable or")
    print(f"place your key in the file: {fallback_path}")
    return None

def get_openai_commit_message(patch):
    """Calls the OpenAI API to generate the commit message."""
    if not OPENAI_AVAILABLE:
        print("OpenAI engine is not available.")
        return None

    api_key = get_api_key("openai")
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key)
        full_prompt = OPENAI_PROMPT_TEMPLATE.format(patch)

        print(f"Requesting commit message from OpenAI ({OPENAI_MODEL})...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You generate Conventional Commit messages based on git diffs."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.5,
            max_tokens=200
        )

        if response.choices and response.choices[0].message:
            message = response.choices[0].message.content.strip()
            if message.startswith("```") and message.endswith("```"):
                message = message[3:-3].strip()
            return message
        else:
            print("Error: Unexpected response from the OpenAI API.")
            # print(response) # Uncomment for detailed response debugging
            return None

    except OpenAIError as e:
        print(f"Error calling the OpenAI API: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during OpenAI interaction: {e}")
        return None

def get_gemini_commit_message(patch):
    """Calls the Gemini API to generate the commit message (Placeholder)."""
    if not GEMINI_AVAILABLE:
        print("Gemini engine is not available.")
        return None

    api_key = get_api_key("gemini")
    if not api_key:
        return None

    try:
        # --- Add your Gemini API call logic here ---
        # Example using google-generativeai:
        # genai.configure(api_key=api_key)
        # model = genai.GenerativeModel(GEMINI_MODEL)
        # full_prompt = GEMINI_PROMPT_TEMPLATE.format(patch)
        # response = model.generate_content(full_prompt)
        # message = response.text.strip()
        # return message
        # --- End of Gemini API call logic ---

        print(f"Requesting commit message from Gemini ({GEMINI_MODEL})...")
        # Placeholder response
        # Replace this with your actual API call
        print("INFO: Gemini API call placeholder executed.")
        return f"feat: Placeholder commit message from Gemini based on patch\n\n{patch[:50]}..." # Simulate a response

    except Exception as e:
        print(f"Error calling the Gemini API: {e}")
        return None

def get_claude_commit_message(patch):
    """Calls the Claude API to generate the commit message (Placeholder)."""
    if not CLAUDE_AVAILABLE:
        print("Claude engine is not available.")
        return None

    api_key = get_api_key("claude")
    if not api_key:
        return None

    try:
        # --- Add your Claude API call logic here ---
        # Example using anthropic:
        # client = Anthropic(api_key=api_key)
        # full_prompt = CLAUDE_PROMPT_TEMPLATE.format(patch)
        # message = client.messages.create(
        #     model=CLAUDE_MODEL,
        #     max_tokens=200,
        #     messages=[
        #         {"role": "user", "content": full_prompt}
        #     ]
        # ).content[0].text.strip()
        # return message
        # --- End of Claude API call logic ---

        print(f"Requesting commit message from Claude ({CLAUDE_MODEL})...")
        # Placeholder response
        # Replace this with your actual API call
        print("INFO: Claude API call placeholder executed.")
        return f"fix: Placeholder commit message from Claude\n\nDetails about the fix based on {patch[:50]}..." # Simulate a response

    except Exception as e:
        print(f"Error calling the Claude API: {e}")
        return None


def save_message_to_file(message, engine_name):
    """Saves the commit message to a timestamped file with engine name."""
    try:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        # Modified filename format to include engine name
        filename = f"commit_msg_{engine_name.lower()}_{timestamp}.txt"
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
    parser = argparse.ArgumentParser(description="Generate a Git commit message using an AI model.")
    parser.add_argument(
        '-a', '--ai-engine',
        choices=['gemini', 'claude', 'openai'],
        default=DEFAULT_ENGINE,
        help=f"Choose the AI engine to use (default: {DEFAULT_ENGINE})"
    )
    args = parser.parse_args()
    chosen_engine = args.ai_engine.lower()

    # Check if the chosen engine is available
    if chosen_engine == 'openai' and not OPENAI_AVAILABLE:
        print("Error: OpenAI engine requested but library is not available. Exiting.")
        sys.exit(1)
    elif chosen_engine == 'gemini' and not GEMINI_AVAILABLE:
        print("Error: Gemini engine requested but library is not available. Exiting.")
        sys.exit(1)
    elif chosen_engine == 'claude' and not CLAUDE_AVAILABLE:
        print("Error: Claude engine requested but library is not available. Exiting.")
        sys.exit(1)


    print("Checking Git repository...")
    if not is_git_repository():
        print("Error: This directory does not appear to be a Git repository.")
        sys.exit(1)
    print("OK.")

    print("Retrieving staged changes (git diff --cached)...")
    patch = get_staged_diff()

    if patch is None:
        sys.exit(1)

    if not patch:
        print("No changes staged for commit (did you forget to 'git add'?).")
        print("Nothing to commit.")
        sys.exit(0)
    print("OK. Found staged changes.")

    # Dictionary to store messages from different engines
    generated_messages = {}
    primary_message = None

    # Determine which engines to call based on the chosen engine
    engines_to_call = [chosen_engine]
    if chosen_engine in ['gemini', 'claude']:
        # Add the other two engines if gemini or claude is chosen
        if 'openai' not in engines_to_call and OPENAI_AVAILABLE:
             engines_to_call.append('openai')
        if 'gemini' not in engines_to_call and GEMINI_AVAILABLE:
             engines_to_call.append('gemini')
        if 'claude' not in engines_to_call and CLAUDE_AVAILABLE:
             engines_to_call.append('claude')
        # Ensure the chosen engine is called first if it's not already
        if engines_to_call[0] != chosen_engine:
             engines_to_call.remove(chosen_engine)
             engines_to_call.insert(0, chosen_engine)


    print(f"\nCalling AI engines: {', '.join(engines_to_call)}")

    # Call the selected AI engines
    for engine in engines_to_call:
        message = None
        if engine == 'openai':
            message = get_openai_commit_message(patch)
        elif engine == 'gemini':
            message = get_gemini_commit_message(patch)
        elif engine == 'claude':
            message = get_claude_commit_message(patch)

        if message:
            generated_messages[engine] = message
            if engine == chosen_engine:
                primary_message = message # Store the message from the primary engine

    if not generated_messages:
        print("\nCould not generate commit message from any available engine. Aborting.")
        sys.exit(1)

    # Display generated messages
    print("\n" + "="*20 + " Suggested Commit Messages " + "="*20)
    for engine, message in generated_messages.items():
        print(f"\n--- Message from {engine.capitalize()} ---")
        print(message)
        print("-" * (20 + len(f" Message from {engine.capitalize()} ") + 2))
    print("=" * (42 + len(" Suggested Commit Messages ")) + "\n")

    # Ask the user for confirmation with the new 'save' option
    # The prompt refers to the message from the primarily chosen engine
    if primary_message:
        print(f"The primary message generated by {chosen_engine.capitalize()} is shown above.")
        while True:
            confirm = input(f"Use the {chosen_engine.capitalize()} message? (y/n/s[ave]): ").lower().strip()
            if confirm == 'y':
                # Proceed with commit using the primary message
                commit_message_to_use = primary_message
                break
            elif confirm == 'n':
                # Cancel commit
                print("Commit cancelled by user.")
                sys.exit(0)
            elif confirm == 's':
                # Save the primary message to a file and exit
                if save_message_to_file(primary_message, chosen_engine):
                    print(f"Exiting without committing. Primary message from {chosen_engine.capitalize()} saved.")
                else:
                    print("Exiting despite save error.")
                sys.exit(0)
            else:
                print("Invalid input. Please enter 'y' (yes), 'n' (no), or 's' (save).")
    else:
        # This case should ideally not be reached if generated_messages is not empty
        # but the primary_message was not set (e.g., if the chosen engine failed).
        # In this scenario, we can't offer to commit a specific message easily.
        print("Could not retrieve a message from the primary engine.")
        # Offer to save all generated messages if any exist, or just exit.
        if generated_messages:
             print("Messages from other engines were generated.")
             save_all = input("Save all generated messages? (y/n): ").lower().strip()
             if save_all == 'y':
                  for engine, msg in generated_messages.items():
                       save_message_to_file(msg, engine)
                  print("Exiting after saving messages.")
                  sys.exit(0)
             else:
                  print("Exiting without saving.")
                  sys.exit(0)
        else:
             # Should have exited earlier if no messages were generated
             sys.exit(1)


    # Execute the commit (only reached if user entered 'y' for the primary message)
    print("\nExecuting git commit...")
    commit_result = run_git_command(['git', 'commit', '-m', commit_message_to_use])

    if commit_result is not None:
        print("\nCommit successful!")
        print("-" * 10 + " Git Output " + "-" * 10)
        print(commit_result)
        print("-" * (20 + len(" Git Output ")))
    else:
        print("\nError during commit execution.")
        print("You can try copying the suggested message and running 'git commit' manually.")
        sys.exit(1)

if __name__ == "__main__":
    main()
