#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
import sys
import datetime
import argparse
import json # Added for handling JSON responses from Ollama
from pathlib import Path

# --- Import AI Libraries ---
# Attempt to import necessary libraries. Handle missing ones gracefully.
try:
    from openai import OpenAI, OpenAIError
    OPENAI_AVAILABLE = True
except ImportError:
    # print("Warning: The 'openai' library is not installed. OpenAI engine will not be available.")
    # print("To install: pip install openai")
    OPENAI_AVAILABLE = False

try:
    # Assuming google-generativeai for Gemini. You might need a different library.
    # To install: pip install google-generativeai
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    # print("Warning: The 'google-generativeai' library is not installed. Gemini engine will not be available.")
    # print("To install: pip install google-generativeai")
    GEMINI_AVAILABLE = False

try:
    # Assuming anthropic for Claude. You might need a different library or direct HTTP.
    # To install: pip install anthropic
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    # print("Warning: The 'anthropic' library is not installed. Claude engine will not be available.")
    # print("To install: pip install anthropic")
    CLAUDE_AVAILABLE = False

try:
    # Required for Ollama HTTP requests
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    # print("Warning: The 'requests' library is not installed. Ollama engine will not be available.")
    # print("To install: pip install requests")
    OLLAMA_AVAILABLE = False


# --- Configuration ---
# Default AI engine(s)
DEFAULT_ENGINES = ["openai"]

# OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo")
OPENAI_MODEL = "gpt-4o-mini"
# Prompt template for OpenAI
OPENAI_PROMPT_TEMPLATE = """
You are a skilled developer specializing in writing excellent commit messages following the Conventional Commits standard (https://www.conventionalcommits.org/).

Analyze the following git patch and generate an appropriate commit message.

The format MUST be:
<type>[optional scope]: <description>

[optional body explaining the 'what' and 'why' vs. 'how']

[optional footer(s) for BREAKING CHANGE or referencing issues (this MUST be present ONLY if the branch name contains a number like /story/ISSUE-456/fixing-code), e.g., Closes #123]

Common types include: feat, fix, build, chore, ci, docs, style, refactor, perf, test.

Here is the git patch:
```diff
{}
```

Generate ONLY the commit message itself, starting directly with the type. Do not include any introductory phrases like "Here is the commit message:".
"""

# --- Placeholder Configurations for Other Engines ---
# You will need to configure models and prompt templates for Gemini and Claude
GEMINI_MODEL = "gemini-2.0-flash" # Example model name
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


CLAUDE_MODEL = "claude-3-5-haiku-20241022" # Example model name
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

# Ollama Configuration
OLLAMA_DEFAULT_HOST = "localhost"
OLLAMA_DEFAULT_PORT = 11434
OLLAMA_DEFAULT_MODEL = "llama3.2" # You might want to make this configurable too
OLLAMA_PROMPT_TEMPLATE = """
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
    # Ollama does not use API keys in this manner, so we handle it separately
    if engine == 'ollama':
        return True # Indicate that Ollama is configured via host/port

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
    """Calls the Gemini API to generate the commit message."""
    if not GEMINI_AVAILABLE:
        print("Gemini engine is not available.")
        return None

    api_key = get_api_key("gemini")
    if not api_key:
        return None

    try:
        print(f"Requesting commit message from Gemini ({GEMINI_MODEL})...")
        # Example using google-generativeai:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        full_prompt = GEMINI_PROMPT_TEMPLATE.format(patch)
        response = model.generate_content(full_prompt)
        message = response.text.strip()
        return message
        # --- End of Gemini API call logic ---

    except Exception as e:
        print(f"Error calling the Gemini API: {e}")
        return None

def get_claude_commit_message(patch):
    """Calls the Claude API to generate the commit message."""
    if not CLAUDE_AVAILABLE:
        print("Claude engine is not available.")
        return None

    api_key = get_api_key("claude")
    if not api_key:
        return None

    try:
        print(f"Requesting commit message from Claude ({CLAUDE_MODEL})...")
        # Example using anthropic:
        client = Anthropic(api_key=api_key)
        full_prompt = CLAUDE_PROMPT_TEMPLATE.format(patch) # Using Claude's specific template
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000, # Increased max tokens slightly
            messages=[
               {"role": "user", "content": full_prompt}
            ]
        ).content[0].text.strip()
        return message
        # --- End of Claude API call logic ---

    except Exception as e:
        print(f"Error calling the Claude API: {e}")
        return None

def get_ollama_commit_message(patch, host, port):
    """Calls the Ollama API to generate the commit message."""
    if not OLLAMA_AVAILABLE:
        print("Ollama engine is not available (requires 'requests' library).")
        return None

    url = f"http://{host}:{port}/api/generate"
    full_prompt = OLLAMA_PROMPT_TEMPLATE.format(patch)

    data = {
        "model": OLLAMA_DEFAULT_MODEL,
        "prompt": full_prompt,
        "stream": False # We want the full response at once
    }

    try:
        print(f"Requesting commit message from Ollama ({OLLAMA_DEFAULT_MODEL}) at {url}...")
        response = requests.post(url, json=data)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        result = response.json()
        message = result.get("response", "").strip()

        if message:
            # Ollama might include the prompt in the response, try to clean it
            if message.startswith(full_prompt):
                message = message[len(full_prompt):].strip()
            if message.startswith("```") and message.endswith("```"):
                 message = message[3:-3].strip()
            return message
        else:
            print("Error: Unexpected response from the Ollama API (empty response).")
            print(result)
            return None

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama server at {url}.")
        print("Please ensure Ollama is running and the host/port are correct.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling the Ollama API: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON response from Ollama at {url}.")
        print("Received response:", response.text)
        return None
    except Exception as e:
        print(f"Unexpected error during Ollama interaction: {e}")
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
        type=str, # Accept string input
        default=','.join(DEFAULT_ENGINES), # Default as comma-separated string
        help=f"Choose one or more AI engines (comma-separated: gemini,claude,openai,ollama) (default: {','.join(DEFAULT_ENGINES)})"
    )
    parser.add_argument(
        '-oh', '--ollama-host',
        type=str,
        default=OLLAMA_DEFAULT_HOST,
        help=f"Specify the Ollama server host (default: {OLLAMA_DEFAULT_HOST})"
    )
    parser.add_argument(
        '-op', '--ollama-port',
        type=int,
        default=OLLAMA_DEFAULT_PORT,
        help=f"Specify the Ollama server port (default: {OLLAMA_DEFAULT_PORT})"
    )
    args = parser.parse_args()

    # Split the comma-separated input into a list of engines
    requested_engines = [e.strip().lower() for e in args.ai_engine.split(',')]

    # Filter out invalid or unavailable engines
    available_engines = {
        'openai': OPENAI_AVAILABLE,
        'gemini': GEMINI_AVAILABLE,
        'claude': CLAUDE_AVAILABLE,
        'ollama': OLLAMA_AVAILABLE # Added Ollama
    }

    engines_to_call = []
    for engine in requested_engines:
        if engine in available_engines:
            if available_engines[engine]:
                engines_to_call.append(engine)
            else:
                print(f"Warning: Requested engine '{engine}' is not available (library not installed). Skipping.")
        else:
            print(f"Warning: Unknown AI engine '{engine}'. Skipping.")

    # If no valid engines were requested or available, use the default
    if not engines_to_call:
        print(f"No valid or available engines requested. Using default: {','.join(DEFAULT_ENGINES)}")
        for engine in DEFAULT_ENGINES:
             if engine in available_engines and available_engines[engine]:
                  engines_to_call.append(engine)
             else:
                  print(f"Warning: Default engine '{engine}' is not available. Skipping.")

    # Final check if any engines can be called
    if not engines_to_call:
         print("Error: No AI engines are available. Please install at least one library (openai, google-generativeai, anthropic, or requests).")
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

    # Dictionary to store messages from different engines {engine_name: message}
    generated_messages = {}

    print(f"\nCalling AI engines: {', '.join(engines_to_call)}")

    # Call the selected AI engines and store results
    for engine in engines_to_call:
        message = None
        # Check for API key or Ollama host/port before calling
        if engine == 'ollama':
            # Ollama doesn't use get_api_key in the same way, check requests availability
            if OLLAMA_AVAILABLE:
                 message = get_ollama_commit_message(patch, args.ollama_host, args.ollama_port)
        else:
            # For other engines, check for API key
            if get_api_key(engine): # get_api_key prints error if not found
                if engine == 'openai':
                    message = get_openai_commit_message(patch)
                elif engine == 'gemini':
                    message = get_gemini_commit_message(patch)
                elif engine == 'claude':
                    message = get_claude_commit_message(patch)

        if message:
            generated_messages[engine] = message

    if not generated_messages:
        print("\nCould not generate commit message from any available engine. Aborting.")
        sys.exit(1)

    # Display generated messages with numbers
    print("\n" + "="*20 + " Suggested Commit Messages " + "="*20)
    message_options = list(generated_messages.items()) # List of (engine, message) tuples
    for i, (engine, message) in enumerate(message_options):
        print(f"\n--- Option {i+1}: Message from {engine.capitalize()} ---")
        print(message)
        print("-" * (len(f" Option {i+1}: Message from {engine.capitalize()} ") + 8)) # Adjust separator length

    print("=" * (42 + len(" Suggested Commit Messages ")) + "\n")

    selected_message_index = 0
    if len(message_options) > 1:
        # Ask the user to select a message
        selected_message_index = -1
        while selected_message_index < 0 or selected_message_index >= len(message_options):
            try:
                selection_input = input(f"Enter the number of the message you want to use (1-{len(message_options)}): ").strip()
                selected_message_index = int(selection_input) - 1 # Convert to 0-based index
                if selected_message_index < 0 or selected_message_index >= len(message_options):
                     print("Invalid number. Please enter a number within the range.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except EOFError: # Handle Ctrl+D
                print("\nInput cancelled. Exiting.")
                sys.exit(0)


    # Get the selected message and its engine name
    selected_engine, commit_message_to_use = message_options[selected_message_index]

    # Ask the user for confirmation (y/n/s) for the selected message
    while True:
        confirm = input(f"Use the selected message from {selected_engine.capitalize()}? (y/n/s[ave]): ").lower().strip()
        if confirm == 'y':
            # Proceed with commit using the selected message
            break
        elif confirm == 'n':
            # Cancel commit
            print("Commit cancelled by user.")
            sys.exit(0)
        elif confirm == 's':
            # Save the selected message to a file and exit
            if save_message_to_file(commit_message_to_use, selected_engine):
                print(f"Exiting without committing. Selected message from {selected_engine.capitalize()} saved.")
            else:
                print("Exiting despite save error.")
            sys.exit(0)
        else:
            print("Invalid input. Please enter 'y' (yes), 'n' (no), or 's' (save).")

    # Execute the commit (only reached if user entered 'y')
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
