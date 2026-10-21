from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 配置
# ============================================================

WORKFLOW_ROOT = Path(
    r"C:\Users\eiealxi\OneDrive - Ericsson\Desktop\workflow"
)

DB_DIR = WORKFLOW_ROOT / "video_kb" / "index_data" / "chroma_db"

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

COLLECTION_CANDIDATES = [
    "video_transcripts",
    "video_chunks",
]

# ============================================================
# 初始化
# ============================================================

print("Loading model ...")
model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("Opening Chroma DB ...")
client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = None

for name in COLLECTION_CANDIDATES:
    try:
        collection = client.get_collection(name)
        print(f"Collection found: {name}")
        break
    except Exception:
        pass

if collection is None:
    names = [c.name for c in client.list_collections()]

    raise RuntimeError(
        f"找不到 Collection。\n现有Collection:\n{names}"
    )

print("Ready")


# ============================================================
# GUI
# ============================================================

class SearchApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Video KB Validator")
        self.root.geometry("1400x850")

        self.results = []

        self.build_ui()

    # --------------------------------------------------------

    def build_ui(self):

        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(
            top,
            text="Query:"
        ).pack(side="left")

        self.query_var = tk.StringVar()

        entry = ttk.Entry(
            top,
            textvariable=self.query_var,
            width=80
        )
        entry.pack(
            side="left",
            padx=5
        )

        entry.bind(
            "<Return>",
            self.search
        )

        ttk.Label(
            top,
            text="TopK:"
        ).pack(
            side="left",
            padx=(20, 5)
        )

        self.topk_var = tk.IntVar(
            value=5
        )

        ttk.Spinbox(
            top,
            from_=1,
            to=20,
            width=5,
            textvariable=self.topk_var
        ).pack(side="left")

        ttk.Button(
            top,
            text="Search",
            command=self.search
        ).pack(
            side="left",
            padx=10
        )

        body = ttk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL
        )
        body.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        # -----------------------------
        # 左侧结果
        # -----------------------------

        left_frame = ttk.Frame(body)

        self.result_list = tk.Listbox(
            left_frame,
            font=("Consolas", 10)
        )

        self.result_list.pack(
            fill="both",
            expand=True
        )

        self.result_list.bind(
            "<<ListboxSelect>>",
            self.show_detail
        )

        body.add(
            left_frame,
            weight=1
        )

        # -----------------------------
        # 右侧详情
        # -----------------------------

        right_frame = ttk.Frame(body)

        self.detail = tk.Text(
            right_frame,
            wrap="word",
            font=("Consolas", 10)
        )

        self.detail.pack(
            fill="both",
            expand=True
        )

        body.add(
            right_frame,
            weight=3
        )

    # --------------------------------------------------------

    def search(self, event=None):

        query = self.query_var.get().strip()

        if not query:
            messagebox.showwarning(
                "Warning",
                "请输入查询内容"
            )
            return

        self.result_list.delete(0, tk.END)
        self.detail.delete("1.0", tk.END)

        try:

            embedding = model.encode(
                [query],
                normalize_embeddings=True
            )[0]

            result = collection.query(
                query_embeddings=[
                    embedding.tolist()
                ],
                n_results=self.topk_var.get()
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )
            return

        docs = result["documents"][0]
        metas = result["metadatas"][0]

        distances = result.get(
            "distances",
            [[None] * len(docs)]
        )[0]

        self.results = []

        for idx, (doc, meta, dist) in enumerate(
            zip(
                docs,
                metas,
                distances
            ),
            start=1
        ):

            video = (
                meta.get("video_name")
                or meta.get("video")
                or "Unknown"
            )

            start_time = meta.get(
                "start_time",
                "-"
            )

            end_time = meta.get(
                "end_time",
                "-"
            )

            dist_str = (
                f"{dist:.4f}"
                if dist is not None
                else "-"
            )

            summary = (
                f"[{idx}] "
                f"{video} | "
                f"{start_time}-{end_time} | "
                f"d={dist_str}"
            )

            self.result_list.insert(
                tk.END,
                summary
            )

            self.results.append(
                {
                    "doc": doc,
                    "meta": meta,
                    "dist": dist,
                }
            )

    # --------------------------------------------------------

    def show_detail(self, event=None):

        sel = self.result_list.curselection()

        if not sel:
            return

        idx = sel[0]

        item = self.results[idx]

        meta = item["meta"]

        self.detail.delete(
            "1.0",
            tk.END
        )

        self.detail.insert(
            tk.END,
            "=" * 60 + "\n"
        )

        self.detail.insert(
            tk.END,
            "Metadata\n"
        )

        self.detail.insert(
            tk.END,
            "=" * 60 + "\n\n"
        )

        for k, v in meta.items():

            self.detail.insert(
                tk.END,
                f"{k}: {v}\n"
            )

        self.detail.insert(
            tk.END,
            "\n\n"
        )

        self.detail.insert(
            tk.END,
            "=" * 60 + "\n"
        )

        self.detail.insert(
            tk.END,
            "Chunk Text\n"
        )

        self.detail.insert(
            tk.END,
            "=" * 60 + "\n\n"
        )

        self.detail.insert(
            tk.END,
            item["doc"]
        )


# ============================================================
# 启动
# ============================================================

root = tk.Tk()
app = SearchApp(root)
root.mainloop()