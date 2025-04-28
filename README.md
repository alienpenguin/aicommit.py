
# AICommit  
**Generate Conventional Commit messages automatically using OpenAI, Gemini, Claude, or Ollama models!**

---

## ✨ About  
`aicommit.py` is a smart command-line tool that uses AI models (OpenAI, Google Gemini, Anthropic Claude, or Ollama) to generate meaningful, Conventional Commit-compliant messages based on your staged Git changes (`git diff --cached`).  

It saves you time, improves commit quality, and maintains consistency across your Git history — especially useful in professional, team, and open-source development.

---

## 🚀 Features
- 🔥 **Multi-Engine Support**: Choose between OpenAI, Gemini, Claude, or a locally running Ollama server.
- 🎯 **Conventional Commits**: Always produces commit messages adhering to [Conventional Commits](https://www.conventionalcommits.org/).
- 🛠️ **Automatic Detection**: Detects your staged Git changes and generates commit suggestions.
- 📝 **Multiple Suggestions**: Supports generating multiple commit messages if multiple engines are selected.
- 📂 **Offline Ollama Option**: Use a local LLM through Ollama without internet!
- 🔒 **Environment Variables or Local Files**: Flexible authentication handling for API keys.
- 💾 **Save Commit Messages**: Optionally save generated commit messages to timestamped files.

---

## 📦 Installation  

First, clone the repository:  
```bash
git clone https://github.com/your-username/aicommit.git
cd aicommit
```

Make the script executable (optional):  
```bash
chmod +x aicommit.py
```

Install dependencies according to the AI engines you plan to use:  
```bash
# Install basic required libraries
pip install openai google-generativeai anthropic requests
```

Or selectively:  
- OpenAI: `pip install openai`
- Gemini: `pip install google-generativeai`
- Claude: `pip install anthropic`
- Ollama (local inference): `pip install requests`

---

## 🔑 Setup API Keys  

Set your API keys either as environment variables:  
```bash
export OPENAI_API_KEY=your-openai-api-key
export GEMINI_API_KEY=your-gemini-api-key
export CLAUDE_API_KEY=your-claude-api-key
```

Or create text files:
- `~/.nr/aicommit/openai_api_key`
- `~/.nr/aicommit/gemini_api_key`
- `~/.nr/aicommit/claude_api_key`

*(Each file should contain only the corresponding API key.)*  

> **Note:** Ollama requires no API key but expects the server running at `localhost:11434` (default) or configurable via arguments.

---

## 🛠️ Usage

From inside a Git repository after staging changes (`git add`):  

```bash
python aicommit.py
```

Options:
```bash
python aicommit.py --ai-engine openai
python aicommit.py --ai-engine gemini,openai,claude
python aicommit.py --ai-engine ollama --ollama-host 127.0.0.1 --ollama-port 11434
```

You will be presented with generated commit messages to choose from!

---

## 🧠 How It Works

- Detects your staged (`git diff --cached`) changes.
- Formats a prompt for the selected AI engine(s).
- Sends the prompt to the API or local Ollama server.
- Displays one or multiple generated commit messages.
- Lets you choose a message to use, save, or cancel.
- If confirmed, commits automatically using `git commit -m`.

---

## ⚙️ Supported Engines  

| Engine | Library | API Key Required | Notes |
|:------|:--------|:-----------------|:------|
| OpenAI | `openai` | ✅ | GPT-4o-mini, GPT-4o, GPT-3.5-turbo |
| Gemini | `google-generativeai` | ✅ | Gemini 2.0 models |
| Claude | `anthropic` | ✅ | Claude 3.5 family |
| Ollama | `requests` | 🚫 | Local LLM (e.g., Llama3.2, Mistral) |

---

## 📋 Example

```bash
$ git add myfile.py
$ python aicommit.py --ai-engine openai
Checking Git repository...
OK.
Retrieving staged changes (git diff --cached)...
OK. Found staged changes.

Calling AI engines: openai

==================== Suggested Commit Messages ====================

--- Option 1: Message from Openai ---
feat(parser): add support for new file format

Extend parsing logic to handle the .abc format files.
This improves compatibility with legacy systems.
---------------------------------------------------------------------

Enter the number of the message you want to use (1-1): 1
Use the selected message from Openai? (y/n/s[ave]): y

Executing git commit...
[main d4c3b2a] feat(parser): add support for new file format
 1 file changed, 10 insertions(+)
Commit successful!
```

---

## ❓ FAQ  

**Q:** Does this stage files for me?  
**A:** No, you still need to `git add` files yourself before running the tool.

**Q:** What happens if multiple engines are selected?  
**A:** You will get multiple commit messages and can pick the one you like.

**Q:** Can I run this without internet access?  
**A:** Yes, if you use the Ollama option with a local LLM.

**Q:** Does it support complex multi-line commit bodies and footers (like "Closes #123")?  
**A:** Yes! It will automatically add issue links if your branch name includes something like `/story/ISSUE-456/feature-title`.

---

## 🧹 Future Improvements  

- Add model configuration.
- Support batch commits (multiple commits at once).

---

## 👨‍💻 Contributing

Pull requests are welcome!  
Please open an issue first to discuss your idea.  
Make sure your code follows basic Python style guidelines.

---

## 📄 License
MIT License. See [`LICENSE`](LICENSE) file for details.

---
