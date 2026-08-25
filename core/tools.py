"""
SpaceaxAI - Tool System v3.0
Tool execution engine untuk Agent SpaceAX (File system, Web search, System execution, Python Runner).
Oleh: Thomas Alfareno Ananta Nugraha - ITS Surabaya
"""

import os
import sys
import subprocess
import glob
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, List


class ToolRegistry:
    """Registry alat bantu (tools) yang dapat dipanggil oleh SpaceAX Agent."""
    
    @staticmethod
    def read_file(path: str) -> str:
        """Membaca isi file teks."""
        try:
            if not os.path.exists(path):
                return f"Error: File '{path}' tidak ditemukan."
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(8000)
            return f"--- Contents of {path} ---\n{content}"
        except Exception as e:
            return f"Error reading file {path}: {e}"

    @staticmethod
    def write_file(path: str, content: str) -> str:
        """Menulis teks ke file (overwrite/create)."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Sukses: Berhasil menulis file ke '{path}'."
        except Exception as e:
            return f"Error writing file {path}: {e}"

    @staticmethod
    def list_dir(path: str = ".") -> str:
        """Melihat daftar file dan folder dalam direktori."""
        try:
            items = os.listdir(path)
            res = [f"📂 Directory contents of '{path}':"]
            for item in items[:40]:
                full = os.path.join(path, item)
                item_type = "DIR " if os.path.isdir(full) else "FILE"
                res.append(f"  [{item_type}] {item}")
            return "\n".join(res)
        except Exception as e:
            return f"Error listing directory {path}: {e}"

    @staticmethod
    def run_command(command: str) -> str:
        """Menjalankan perintah terminal/PowerShell/bash."""
        try:
            res = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=15
            )
            stdout = res.stdout[:2000] if res.stdout else ""
            stderr = res.stderr[:1000] if res.stderr else ""
            return f"Return Code: {res.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Command execution timed out (15s limit)."
        except Exception as e:
            return f"Error running command: {e}"

    @staticmethod
    def execute_python(code: str) -> str:
        """Menjalankan snippet kode Python secara independen."""
        try:
            res = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, timeout=10
            )
            return f"Python Output:\n{res.stdout}\nErrors (if any):\n{res.stderr}"
        except Exception as e:
            return f"Error executing Python code: {e}"

    @staticmethod
    def web_search(query: str) -> str:
        """Cari informasi di internet."""
        from learning.internet import InternetLearner
        try:
            learner = InternetLearner("data/knowledge")
            return learner.search_and_learn(query)
        except Exception as e:
            return f"Error web search: {e}"


def get_available_tools_description() -> str:
    """Deskripsi tool untuk prompt Agent."""
    return """
Available Tools for SpaceAX Agent:
1. `read_file(path)` - Read content of a text file
2. `write_file(path, content)` - Write content to a file
3. `list_dir(path)` - List directory contents
4. `run_command(command)` - Run a shell/PowerShell command
5. `execute_python(code)` - Run Python code snippet
6. `web_search(query)` - Search information on the internet
"""
