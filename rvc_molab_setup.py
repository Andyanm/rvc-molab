# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # 在 molab 上搭建 Applio (RVC) AI 翻唱环境

        流程:检测 GPU → 克隆 Applio 源码 → 装依赖 → 下载底模 → 启动 WebUI(生成公网链接)

        **运行前先点右上角"notebook 规格"按钮,把 GPU 开关打开**,否则推理会很慢。
        从上到下依次运行每个单元格(Shift+Enter)。
        """
    )
    return


@app.cell
def _():
    import subprocess

    def run(cmd, cwd=None):
        print("$ " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout[-2000:])
        if result.returncode != 0 and result.stderr:
            print(result.stderr[-2000:])
        return result

    return (run,)


@app.cell
def _(mo):
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
    except ImportError:
        gpu_ok = False
    mo.md(
        f"GPU 可用: **{gpu_ok}**"
        + (" ✅" if gpu_ok else " ⚠️ 请先在右上角开启 GPU,再重新运行这个单元格")
    )
    return (gpu_ok,)


@app.cell
def _(run):
    import os

    REPO_DIR = "/tmp/Applio"
    if not os.path.exists(REPO_DIR):
        run(["git", "clone", "--depth", "1", "https://github.com/IAHispano/Applio.git", REPO_DIR])
    else:
        print("Applio 已存在,跳过克隆")
    return REPO_DIR, os


@app.cell
def _(mo):
    mo.md("### 安装依赖\n第一次运行会比较慢(几分钟),耐心等待。")
    return


@app.cell
def _(REPO_DIR, run):
    # 系统级依赖(部分molab容器可能无 sudo/apt 权限,失败可忽略,继续往下走)
    apt_result = run(["apt-get", "update"])
    if apt_result.returncode == 0:
        run(["apt-get", "install", "-y", "ffmpeg"])
    else:
        print("⚠️ 无法用 apt 安装系统依赖(容器权限受限),若 molab 已预装 ffmpeg 可忽略")

    # Applio 自身 Python 依赖
    pip_result = run(["pip", "install", "-r", f"{REPO_DIR}/requirements.txt"])
    return (pip_result,)


@app.cell
def _(REPO_DIR, mo, run):
    # 下载底模:hubert_base.pt / rmvpe.pt / fcpe.pt,以及 v2 预训练权重
    prereq_result = run(
        [
            "python", "core.py", "prerequisites",
            "--models", "True",
            "--pretraineds_v1", "False",
            "--pretraineds_v2", "True",
            "--exe", "False",
        ],
        cwd=REPO_DIR,
    )
    mo.md("✅ 底模下载完成" if prereq_result.returncode == 0 else "⚠️ 底模下载失败,查看上面输出定位原因")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### 启动 WebUI

        下一个单元格会在后台启动 Applio 的 Gradio 界面,并加 `--share`
        生成一个 72 小时内有效的公网链接(molab 本身不做端口转发,这样最省事)。

        启动后**等 30~60 秒**,再运行最后那个"获取链接"的单元格;如果还没出现链接,
        隔几十秒重新运行一次就行。
        """
    )
    return


@app.cell
def _(REPO_DIR):
    import threading
    import re
    import subprocess as sp

    job_state = {"lines": [], "url": None}

    def _launch():
        proc = sp.Popen(
            ["python", "app.py", "--share"],
            cwd=REPO_DIR,
            stdout=sp.PIPE,
            stderr=sp.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            job_state["lines"].append(line)
            match = re.search(r"(https://[a-zA-Z0-9]+\.gradio\.live)", line)
            if match:
                job_state["url"] = match.group(1)

    threading.Thread(target=_launch, daemon=True).start()
    return (job_state,)


@app.cell
def _(job_state, mo):
    import time
    time.sleep(45)
    if job_state["url"]:
        mo.md(f"🎉 **WebUI 已就绪:** [{job_state['url']}]({job_state['url']})")
    else:
        mo.md("⏳ 还没拿到链接,再等一会重新运行本单元格(Ctrl/Cmd+Enter)")
    return


if __name__ == "__main__":
    app.run()
