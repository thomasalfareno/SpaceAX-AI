"""
SpaceaxAI - Agent System v3.0
Autonomous Agent Loop (Plan -> Tool Call -> Observe -> Resolve)
Membuat SpaceAX AI bertindak seperti Agent cerdas (Claude Code, Hermes, Antigravity).
Oleh: Thomas Alfareno Ananta Nugraha - ITS Surabaya
"""

import re
import json
from typing import Dict, Any, List, Optional
from core.tools import ToolRegistry, get_available_tools_description


class SpaceaxAgent:
    """Agent Loop untuk eksekusi perintah kompleks secara otonom."""

    def __init__(self, chat_engine=None):
        self.chat_engine = chat_engine
        self.max_steps = 6

    def run_agent_loop(self, user_goal: str, callback=None) -> Dict[str, Any]:
        """
        Menjalankan Agent loop untuk mencapai target/jawaban user.
        Args:
            user_goal: Perintah atau instruksi dari user.
            callback: Fungsi callback untuk memantau step (optional).
        Returns:
            Dict berisi hasil akhir, langkah-langkah, dan logs.
        """
        steps_log = []
        current_context = f"Goal: {user_goal}\n"

        for step_idx in range(1, self.max_steps + 1):
            if callback:
                callback(f"🤖 [Agent Step {step_idx}/{self.max_steps}] Menganalisis instruksi...")

            # 1. Cek apakah instruksi butuh panggillan tool spesifik
            tool_call = self._detect_tool_intent(user_goal, current_context)

            if not tool_call:
                # Tidak butuh tool lagi atau goal sudah selesai
                break

            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            if callback:
                callback(f"⚙️ [Tool Action] Panggilan `{tool_name}` dengan argumen: {tool_args}")

            # 2. Eksekusi Tool
            result = self._execute_tool(tool_name, tool_args)
            steps_log.append({
                "step": step_idx,
                "tool": tool_name,
                "args": tool_args,
                "result": result
            })

            # Update context dengan observasi hasil tool
            current_context += f"\nObservation Step {step_idx} ({tool_name}):\n{result[:1500]}\n"

            # Jika file write / python command sukses, mungkin sudah selesai
            if tool_name in ["write_file", "execute_python"] and "Sukses" in result:
                break

        # Final Response Synthesis
        if callback:
            callback("Synthesizing final agent report...")

        final_summary = self._synthesize_final_response(user_goal, steps_log)
        return {
            "goal": user_goal,
            "steps": steps_log,
            "response": final_summary
        }

    def _detect_tool_intent(self, goal: str, context: str) -> Optional[Dict[str, Any]]:
        """Mendeteksi secara heuristik/neural alat yang perlu dipanggil."""
        g = goal.lower()

        # Deteksi Web Search
        if any(w in g for w in ["cari di internet", "search", "siapa itu", "berita terbaru", "info terkini"]):
            clean_q = re.sub(r"^(cari|search|tolong cari|info|siapa)\s+", "", goal, flags=re.IGNORECASE).strip()
            return {"name": "web_search", "args": {"query": clean_q or goal}}

        # Deteksi Baca File
        file_read_match = re.search(r"(baca|buka|isi)\s+file\s+([^\s]+)", goal, re.IGNORECASE)
        if file_read_match:
            return {"name": "read_file", "args": {"path": file_read_match.group(2)}}

        # Deteksi Tulis File
        file_write_match = re.search(r"(buat|tulis|simpan)\s+file\s+([^\s]+)", goal, re.IGNORECASE)
        if file_write_match:
            filepath = file_write_match.group(2)
            return {"name": "write_file", "args": {"path": filepath, "content": f"# Output generated for: {goal}\n"}}

        # Deteksi Jalankan Kode Python
        if "jalankan kode python" in g or "running python" in g or g.startswith("python -c"):
            code = goal.replace("jalankan kode python", "").strip()
            return {"name": "execute_python", "args": {"code": code or "print('Agent execution ok')"}}

        # Deteksi Command Terminal
        cmd_match = re.search(r"(jalankan command|run command|cmd)\s+(.+)", goal, re.IGNORECASE)
        if cmd_match:
            return {"name": "run_command", "args": {"command": cmd_match.group(2)}}

        # Deteksi List Directory
        if "list file" in g or "isi folder" in g or "folder contents" in g:
            return {"name": "list_dir", "args": {"path": "."}}

        return None

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Menjalankan tool yang dipilih dari ToolRegistry."""
        if name == "read_file":
            return ToolRegistry.read_file(args.get("path", ""))
        elif name == "write_file":
            return ToolRegistry.write_file(args.get("path", ""), args.get("content", ""))
        elif name == "list_dir":
            return ToolRegistry.list_dir(args.get("path", "."))
        elif name == "run_command":
            return ToolRegistry.run_command(args.get("command", ""))
        elif name == "execute_python":
            return ToolRegistry.execute_python(args.get("code", ""))
        elif name == "web_search":
            return ToolRegistry.web_search(args.get("query", ""))
        return f"Unknown tool: {name}"

    def _synthesize_final_response(self, goal: str, steps_log: List[Dict]) -> str:
        """Menyusun rangkuman jawaban natural setelah agent beraksi."""
        if not steps_log:
            return f"Task completed: '{goal}'"

        resp = [f"### SpaceAX Agent Summary"]
        resp.append(f"**Goal:** {goal}\n")

        for s in steps_log:
            resp.append(f"**Step {s['step']} (`{s['tool']}`):**")
            res_preview = str(s['result']).strip()
            if len(res_preview) > 300:
                res_preview = res_preview[:300] + "..."
            resp.append(f"```\n{res_preview}\n```\n")

        resp.append("**Status:** Execution finished successfully.")
        return "\n".join(resp)
