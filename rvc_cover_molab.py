import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="RVC AI 翻唱 · molab")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        r"""
        # 🎤 RVC AI 翻唱 · molab 完整版

        基于 **[Applio](https://github.com/IAHispano/Applio)**（RVC v2 维护最活跃的分支）+ **[audio-separator](https://github.com/nomadkaraoke/python-audio-separator)**（UVR / Roformer 人声分离）。

        **使用顺序**：
        1. 右上角 *notebook specs* → 打开 **GPU**
        2. 第 ① 步「安装环境」（首次约 5–10 分钟）
        3. 第 ② 步准备声音模型（链接下载 / 上传 `.pth` + `.index` / 自己训练）
        4. 第 ③–⑥ 步：上传歌曲 → 分离人声 → RVC 转换 → 混音导出
        5. 可选：⑦ 训练自己的模型，⑧ 启动 Applio 完整 WebUI（含 TTS、模型融合、实时变声等）

        > ⚠️ molab 仅持久化「侧边栏上传的文件」，生成的模型和音频请及时下载。请只使用你有权使用的声音与歌曲。
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _():
    # ============ 全局路径与工具函数 ============
    import io, os, sys, re, json, glob, shutil, subprocess, zipfile, time
    from pathlib import Path

    ROOT = Path(os.environ.get("RVC_ROOT", str(Path.home() / "rvc_work"))).resolve()
    APPLIO = ROOT / "Applio"
    VENV = ROOT / "venv"
    VPY = VENV / "bin" / "python"
    INPUT_DIR = ROOT / "inputs"
    OUT_DIR = ROOT / "outputs"
    SEP_MODELS = ROOT / "sep_models"
    DATASETS = ROOT / "datasets"
    EXPORTS = ROOT / "exports"
    TOOL = ROOT / "cover_tool.py"
    LOGS = APPLIO / "logs"
    APPLIO_REPO = "https://github.com/IAHispano/Applio.git"
    APPLIO_REF = "939d9ede94d563eb5b96a55dc3922e03f93d4064"  # 已核对 CLI 参数的版本；改成 "main" 可跟随最新
    for _d in (ROOT, INPUT_DIR, OUT_DIR, SEP_MODELS, DATASETS, EXPORTS):
        _d.mkdir(parents=True, exist_ok=True)

    def venv_env():
        env = os.environ.copy()
        _ff = VENV / "ffbin"
        env["PATH"] = f"{_ff}:{VENV / 'bin'}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(VENV)
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("PYTHONPATH", None)
        return env

    def sh(cmd, cwd=None, env=None, check=True, quiet_re=None):
        """运行命令并实时打印输出；返回 (returncode, 最后 200 行)."""
        if isinstance(cmd, (list, tuple)):
            cmd = [str(c) for c in cmd]
            print("$", " ".join(cmd)[:400], flush=True)
        else:
            print("$", cmd[:400], flush=True)
        p = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None, env=env or venv_env(),
            shell=isinstance(cmd, str), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        tail = []
        for line in p.stdout:
            line = line.rstrip()
            tail.append(line)
            tail[:] = tail[-200:]
            if quiet_re and re.search(quiet_re, line):
                continue
            print(line, flush=True)
        p.wait()
        if check and p.returncode != 0:
            raise RuntimeError(f"命令失败（exit {p.returncode}）：\n" + "\n".join(tail[-30:]))
        return p.returncode, tail

    def is_installed():
        return VPY.exists() and (APPLIO / "core.py").exists() and (ROOT / ".install_ok").exists()

    def list_models():
        out = {}
        if LOGS.exists():
            for d in sorted(LOGS.iterdir()):
                if not d.is_dir() or d.name in ("mute", "mute_spin", "mute_spin-v2", "reference", "zips"):
                    continue
                pths = sorted(d.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
                pths = [p for p in pths if not re.match(r"^(G|D)_", p.name)]
                if pths:
                    idx = sorted(d.glob("*.index"), key=lambda p: p.stat().st_mtime, reverse=True)
                    out[d.name] = (str(pths[0]), str(idx[0]) if idx else "")
        return out

    def safe_name(s):
        s = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", s).strip("_")
        return s[:60] or "item"

    def file_dl(path, label=None):
        import marimo as _mo
        path = Path(path)
        return _mo.download(data=lambda: path.read_bytes(), filename=path.name,
                            label=label or f"⬇️ {path.name}")
    return (
        APPLIO, APPLIO_REF, APPLIO_REPO, DATASETS, EXPORTS, INPUT_DIR, LOGS,
        OUT_DIR, io, Path, ROOT, SEP_MODELS, TOOL, VENV, VPY, file_dl, glob,
        is_installed, json, list_models, os, re, safe_name, sh, shutil,
        subprocess, sys, time, venv_env, zipfile,
    )


@app.cell(hide_code=True)
def _():
    # ============ 辅助脚本：分离 / 下载 / 混音（在独立 venv 中运行）============
    COVER_TOOL_SRC = r'''
import argparse, os, sys, json, shutil, subprocess, re

PRIMARY = ["vocals", "noreverb", "no reverb", "dry", "no echo", "no noise"]
SECONDARY = ["instrumental", "other", "reverb", "no dry", "echo", "no vocals", "noise"]


def separate(inp, model, out_dir, prefix, model_dir, seg=None):
    from audio_separator.separator import Separator
    os.makedirs(out_dir, exist_ok=True)
    names = {k: f"{prefix}_main" for k in PRIMARY}
    names.update({k: f"{prefix}_rest" for k in SECONDARY})
    kw = {}
    if seg:
        kw["mdxc_params"] = {"segment_size": int(seg), "override_model_segment_size": False,
                             "batch_size": 1, "overlap": 8, "pitch_shift": 0}
    sep = Separator(model_file_dir=model_dir, output_dir=out_dir, output_format="WAV",
                    use_autocast=True, sample_rate=44100, **kw)
    sep.load_model(model_filename=model)
    files = sep.separate(inp, custom_output_names=names)
    main = os.path.join(out_dir, f"{prefix}_main.wav")
    rest = os.path.join(out_dir, f"{prefix}_rest.wav")
    if not (os.path.exists(main) and os.path.exists(rest)):
        files = [f if os.path.isabs(f) else os.path.join(out_dir, f) for f in files]
        for f in files:
            m = re.search(r"_\(([^)]*)\)_", os.path.basename(f))
            stem = (m.group(1).lower() if m else "")
            if stem in PRIMARY and not os.path.exists(main):
                shutil.move(f, main)
            elif not os.path.exists(rest):
                shutil.move(f, rest)
    print("RESULT_JSON=" + json.dumps({"main": main, "rest": rest}, ensure_ascii=False))


def fetch(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tpl = os.path.join(out_dir, "%(title).60s.%(ext)s")
    cmd = [sys.executable, "-m", "yt_dlp", "-x", "--audio-format", "wav", "--no-playlist",
           "-o", tpl, "--print", "after_move:filepath", url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(r.stderr[-3000:])
    if r.returncode != 0:
        sys.exit(r.returncode)
    print("RESULT_JSON=" + json.dumps({"path": r.stdout.strip().splitlines()[-1]}, ensure_ascii=False))


def _load(path, sr=44100):
    import librosa, numpy as np
    y, _ = librosa.load(path, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.astype("float32")


def _fit(a, n):
    import numpy as np
    return a[:, :n] if a.shape[1] >= n else np.pad(a, ((0, 0), (0, n - a.shape[1])))


def mix(a):
    import numpy as np, soundfile as sf
    from pedalboard import (Pedalboard, Reverb, Compressor, HighpassFilter,
                            PitchShift, Limiter, Gain)
    sr = 44100
    lead = _load(a.lead, sr)
    fx = [HighpassFilter(cutoff_frequency_hz=80)]
    if a.compress:
        fx.append(Compressor(threshold_db=-18, ratio=3, attack_ms=5, release_ms=120))
    if a.reverb > 0:
        fx.append(Reverb(room_size=a.room, wet_level=a.reverb, dry_level=1.0 - a.reverb * 0.4, width=1.0))
    fx.append(Gain(gain_db=a.lead_db))
    layers = [Pedalboard(fx)(lead, sr)]
    for path, db, rev in ((a.backing, a.backing_db, True), (a.inst, a.inst_db, False)):
        if path and os.path.exists(path):
            y = _load(path, sr)
            chain = []
            if a.inst_pitch != 0:
                chain.append(PitchShift(semitones=a.inst_pitch))
            if rev and a.reverb > 0:
                chain.append(Reverb(room_size=a.room, wet_level=a.reverb * 0.8, dry_level=1.0, width=1.0))
            chain.append(Gain(gain_db=db))
            layers.append(Pedalboard(chain)(y, sr))
    n = max(l.shape[1] for l in layers)
    out = sum(_fit(l, n) for l in layers).astype("float32")
    out = Pedalboard([Limiter(threshold_db=-1.0, release_ms=100)])(out, sr)
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.99:
        out = out / peak * 0.98
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    wav = os.path.splitext(a.out)[0] + ".wav"
    sf.write(wav, out.T, sr)
    res = {"wav": wav}
    ff = shutil.which("ffmpeg")
    if ff:
        mp3 = os.path.splitext(a.out)[0] + ".mp3"
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", wav, "-b:a", "320k", mp3], check=True)
        res["mp3"] = mp3
    print("RESULT_JSON=" + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("separate")
    for k in ("--input", "--model", "--out-dir", "--prefix", "--model-dir"):
        s.add_argument(k, required=True)
    s.add_argument("--seg", default=None)
    f = sp.add_parser("fetch"); f.add_argument("--url", required=True); f.add_argument("--out-dir", required=True)
    m = sp.add_parser("mix")
    m.add_argument("--lead", required=True); m.add_argument("--inst"); m.add_argument("--backing")
    m.add_argument("--out", required=True)
    for k, d in (("--lead-db", 0.0), ("--inst-db", 0.0), ("--backing-db", -3.0),
                 ("--inst-pitch", 0.0), ("--reverb", 0.15), ("--room", 0.3)):
        m.add_argument(k, type=float, default=d)
    m.add_argument("--compress", action="store_true")
    a = p.parse_args()
    if a.cmd == "separate":
        separate(a.input, a.model, a.out_dir, a.prefix, a.model_dir, a.seg)
    elif a.cmd == "fetch":
        fetch(a.url, a.out_dir)
    else:
        mix(a)
'''

    SD_STUB = r'''
# sounddevice 占位模块：云端没有声卡/PortAudio，仅用于让 Applio WebUI 正常启动（实时变声页不可用）
class PortAudioError(Exception):
    pass
class _Dummy:
    def __init__(self, *a, **k):
        raise PortAudioError("云端环境无音频设备")
AsioSettings = WasapiSettings = InputStream = OutputStream = Stream = _Dummy
def query_devices(*a, **k):
    raise PortAudioError("云端环境无音频设备")
def query_hostapis(*a, **k):
    raise PortAudioError("云端环境无音频设备")
def _terminate():
    pass
def _initialize():
    pass
default = type("default", (), {"device": (None, None), "samplerate": None})()
'''
    return COVER_TOOL_SRC, SD_STUB


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## ① 安装环境""")
    return


@app.cell(hide_code=True)
def _(mo):
    install_btn = mo.ui.run_button(label="🚀 安装 / 修复环境", kind="success")
    reinstall_chk = mo.ui.checkbox(label="强制重装（删除旧 venv）")
    mo.hstack([install_btn, reinstall_chk], justify="start")
    return install_btn, reinstall_chk


@app.cell
def _(
    APPLIO, APPLIO_REF, APPLIO_REPO, COVER_TOOL_SRC, ROOT, SD_STUB, TOOL,
    VENV, VPY, install_btn, is_installed, mo, os, reinstall_chk, sh, shutil,
    subprocess, sys,
):
    mo.stop(not install_btn.value,
            mo.callout(mo.md("✅ 环境已安装，可直接往下走" if is_installed()
                             else "点击上方按钮开始安装（需先打开 GPU）"),
                       kind="success" if is_installed() else "warn"))

    # 0) GPU 信息
    sh("nvidia-smi || echo '⚠️ 未检测到 GPU：请在右上角 notebook specs 打开 GPU'", env=os.environ.copy(), check=False)

    # 1) uv
    _uv = shutil.which("uv")
    if not _uv:
        sh([sys.executable, "-m", "pip", "install", "-q", "uv"], env=os.environ.copy())
        _uv = shutil.which("uv") or os.path.join(os.path.dirname(sys.executable), "uv")
    UV = [_uv] if os.path.exists(_uv) else [sys.executable, "-m", "uv"]

    # 2) 可选系统包（有 sudo/root 才装；没有也不影响）
    sh("(command -v apt-get >/dev/null && (sudo -n true 2>/dev/null && S=sudo || S=''); "
       "$S apt-get -qq update && $S apt-get -qq install -y ffmpeg libportaudio2 >/dev/null) "
       "|| echo '跳过 apt（无权限），将使用 pip 版 ffmpeg'", env=os.environ.copy(), check=False)

    # 3) Applio 源码
    if reinstall_chk.value and VENV.exists():
        shutil.rmtree(VENV)
    if not (APPLIO / ".git").exists():
        sh(["git", "clone", "-q", APPLIO_REPO, APPLIO], env=os.environ.copy())
    sh(["git", "-c", "advice.detachedHead=false", "fetch", "-q", "origin"], cwd=APPLIO, env=os.environ.copy(), check=False)
    sh(["git", "-c", "advice.detachedHead=false", "checkout", "-q", APPLIO_REF], cwd=APPLIO, env=os.environ.copy())

    # 4) 独立 Python 3.12 venv（与 marimo 内核隔离，避免依赖冲突）
    if not VPY.exists():
        sh(UV + ["venv", str(VENV), "--python", "3.12", "--seed", "-q"], env=os.environ.copy())
    _idx = ["--extra-index-url", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match"]
    _pip = UV + ["pip", "install", "--python", str(VPY)]
    sh(_pip + ["-r", str(APPLIO / "requirements.txt"), "torchvision==0.26.0"] + _idx, quiet_re=r"^\s*[+-~] ")
    # audio-separator：不装 diffq（仅 Demucs 需要且要编译），其余依赖显式安装
    sh(_pip + ["--no-deps", "audio-separator==0.47.0"])
    sh(_pip + ["beartype==0.18.5", "julius", "ml_collections", "onnx-weekly", "onnx2torch-py313",
               "onnxruntime-gpu", "pydub", "pyyaml", "resampy", "rotary-embedding-torch>=0.6.1,<0.7.0",
               "samplerate==0.1.0", "six", "audioread", "yt-dlp", "static-ffmpeg", "huggingface_hub"] + _idx,
       quiet_re=r"^\s*[+-~] ")

    # 5) ffmpeg：优先系统，否则用 static-ffmpeg 并链接到 venv/ffbin
    _ffbin = VENV / "ffbin"
    _ffbin.mkdir(exist_ok=True)
    if not shutil.which("ffmpeg"):
        _r = subprocess.run([str(VPY), "-c", "import static_ffmpeg.run as r;print('\\n'.join(r.get_or_fetch_platform_executables_else_raise()))"],
                            capture_output=True, text=True)
        for _p in _r.stdout.split():
            _dst = _ffbin / os.path.basename(_p)
            if not _dst.exists():
                os.symlink(_p, _dst)

    # 6) 没有 PortAudio 时放一个 sounddevice 占位模块，保证 WebUI 可启动
    _ok = subprocess.run([str(VPY), "-c", "import sounddevice"], capture_output=True).returncode == 0
    if not _ok:
        _site = subprocess.run([str(VPY), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                               capture_output=True, text=True).stdout.strip()
        with open(os.path.join(_site, "sounddevice.py"), "w") as _f:
            _f.write(SD_STUB)
        import glob as _g
        for _d in _g.glob(os.path.join(_site, "sounddevice*.dist-info")):
            shutil.rmtree(_d, ignore_errors=True)
        print("已安装 sounddevice 占位模块（云端无声卡，实时变声不可用，其余功能正常）")

    # 7) Applio 配置 + 预训练/基础模型（rmvpe、fcpe、contentvec、HiFi-GAN/RefineGAN 预训练，约 2 GB）
    if not (APPLIO / "assets" / "config.json").exists():
        shutil.copy(APPLIO / "assets" / "config_template.json", APPLIO / "assets" / "config.json")
    sh([VPY, "core.py", "prerequisites", "--models", "--pretraineds-hifigan", "--no-exe"], cwd=APPLIO,
       quiet_re=r"Downloading all files:")

    TOOL.write_text(COVER_TOOL_SRC, encoding="utf-8")

    # 8) 自检
    sh([VPY, "-c", "import torch;print('torch',torch.__version__,'CUDA',torch.cuda.is_available(),"
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '');"
        "from transformers import HubertModel;import audio_separator.separator, faiss, pedalboard;print('imports OK')"],
       cwd=APPLIO)
    (ROOT / ".install_ok").write_text("ok")
    mo.callout(mo.md("### ✅ 安装完成！继续第 ② 步"), kind="success")
    return


@app.cell(hide_code=True)
def _(COVER_TOOL_SRC, TOOL, is_installed):
    # 每次打开都刷新辅助脚本（环境已装时）
    if is_installed():
        TOOL.write_text(COVER_TOOL_SRC, encoding="utf-8")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## ② 声音模型

        - **链接下载**：支持 HuggingFace（`.zip` 直链、`/resolve/…` 文件、或整个仓库 `https://huggingface.co/用户/仓库`）、Google Drive、其他 zip 直链
        - **上传**：`.zip`（内含 pth+index），或同时选中 `.pth` 和 `.index`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    get_model_ver, set_model_ver = mo.state(0)
    return get_model_ver, set_model_ver


@app.cell(hide_code=True)
def _(mo):
    model_url = mo.ui.text(label="模型链接", placeholder="https://huggingface.co/xxx/yyy 或 .zip 链接", full_width=True)
    model_name_in = mo.ui.text(label="保存名称（可空，自动取名）", placeholder="例如 MyVoice")
    model_dl_btn = mo.ui.run_button(label="⬇️ 下载模型")
    model_upload = mo.ui.file(filetypes=[".zip", ".pth", ".index"], multiple=True, kind="area",
                              max_size=3_000_000_000, label="拖入 .zip 或 .pth + .index")
    model_up_btn = mo.ui.run_button(label="📥 导入上传的模型")
    mo.vstack([model_url, mo.hstack([model_name_in, model_dl_btn], justify="start"),
               model_upload, model_up_btn])
    return model_dl_btn, model_name_in, model_up_btn, model_upload, model_url


@app.cell
def _(
    APPLIO, LOGS, VPY, get_model_ver, is_installed, json, mo, model_dl_btn,
    model_name_in, model_url, re, safe_name, set_model_ver, sh, shutil,
):
    mo.stop(not model_dl_btn.value)
    mo.stop(not is_installed(), mo.callout("请先完成第 ① 步安装", kind="danger"))
    _url = model_url.value.strip()
    mo.stop(not _url, mo.callout("请填写链接", kind="warn"))

    _m = re.match(r"^https?://huggingface\.co/([^/]+/[^/?#]+)/?(?:tree/[^/]+/?)?$", _url)
    if _m:
        # 整个 HF 仓库：挑 .pth / .index / .zip 下载
        _repo = _m.group(1)
        _name = safe_name(model_name_in.value or _repo.split("/")[-1])
        _code = f"""
import json, os, shutil
from huggingface_hub import HfApi, hf_hub_download
repo = {json.dumps(_repo)}
files = [f for f in HfApi().list_repo_files(repo) if f.lower().endswith(('.pth', '.index', '.zip'))]
print('仓库文件:', files)
dst = {json.dumps(str(LOGS / _name))}
os.makedirs(dst, exist_ok=True)
pths = [f for f in files if f.endswith('.pth') and not os.path.basename(f).startswith(('G_', 'D_'))]
idx = [f for f in files if f.endswith('.index')]
zips = [f for f in files if f.endswith('.zip')]
if pths:
    for f in pths[:1] + idx[:1]:
        p = hf_hub_download(repo, f)
        ext = os.path.splitext(f)[1]
        shutil.copy(p, os.path.join(dst, {json.dumps(_name)} + ext))
        print('✔', f)
elif zips:
    import zipfile
    p = hf_hub_download(repo, zips[0])
    zipfile.ZipFile(p).extractall(dst)
    print('✔ 解压', zips[0])
else:
    raise SystemExit('仓库里没找到 .pth/.index/.zip')
"""
        sh([VPY, "-c", _code], cwd=APPLIO)
        # 若 zip 解压后在子目录中，拍平
        for _f in list((LOGS / _name).rglob("*")):
            if _f.is_file() and _f.suffix in (".pth", ".index") and _f.parent != LOGS / _name:
                shutil.move(str(_f), str(LOGS / _name / _f.name))
    else:
        sh([VPY, "core.py", "download", "--model-link", _url], cwd=APPLIO)
    set_model_ver(get_model_ver() + 1)
    mo.callout("✅ 模型下载完成，请在下方选择模型", kind="success")
    return


@app.cell
def _(
    LOGS, get_model_ver, io, mo, model_name_in, model_up_btn, model_upload,
    safe_name, set_model_ver, shutil, zipfile,
):
    mo.stop(not model_up_btn.value)
    mo.stop(not model_upload.value, mo.callout("请先选择文件", kind="warn"))
    _files = model_upload.value
    _base = model_name_in.value or [f.name for f in _files if f.name.endswith((".pth", ".zip"))][0].rsplit(".", 1)[0]
    _name = safe_name(_base)
    _dst = LOGS / _name
    _dst.mkdir(parents=True, exist_ok=True)
    for _f in _files:
        if _f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(_f.contents)) as _z:
                for _member in _z.namelist():
                    if _member.lower().endswith((".pth", ".index")) and not _member.startswith("__MACOSX"):
                        _ext = _member.rsplit(".", 1)[1]
                        _fn = _member.split("/")[-1]
                        _target = _dst / (_fn if _fn.startswith(("G_", "D_")) else f"{_name}.{_ext}" if _ext == "index" else _fn)
                        with _z.open(_member) as _src, open(_target, "wb") as _out:
                            shutil.copyfileobj(_src, _out)
        else:
            _ext = _f.name.rsplit(".", 1)[1].lower()
            (_dst / f"{_name}.{_ext}").write_bytes(_f.contents)
    set_model_ver(get_model_ver() + 1)
    mo.callout(f"✅ 已导入模型 **{_name}**", kind="success")
    return


@app.cell(hide_code=True)
def _(get_model_ver, list_models, mo):
    get_model_ver()
    MODELS = list_models()
    voice_model = mo.ui.dropdown(options=list(MODELS.keys()), value=(list(MODELS.keys()) or [None])[0],
                                 label="🎙️ 选择声音模型", searchable=True)
    mo.vstack([voice_model,
               mo.md(f"已有 **{len(MODELS)}** 个模型" if MODELS else "_还没有模型，请先下载/上传/训练_")])
    return MODELS, voice_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## ③ 输入歌曲""")
    return


@app.cell(hide_code=True)
def _(mo):
    song_upload = mo.ui.file(filetypes=[".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".opus", ".webm", ".mp4"],
                             kind="area", max_size=500_000_000, label="拖入歌曲文件")
    song_url = mo.ui.text(label="或粘贴链接（B站/YouTube/SoundCloud 等，yt-dlp 支持即可）", full_width=True)
    song_btn = mo.ui.run_button(label="📀 载入歌曲")
    mo.vstack([song_upload, song_url, song_btn])
    return song_btn, song_upload, song_url


@app.cell
def _(
    INPUT_DIR, OUT_DIR, Path, TOOL, VPY, json, mo, safe_name, sh, song_btn,
    song_upload, song_url,
):
    mo.stop(not song_btn.value, mo.md("_上传或填链接后点「载入歌曲」_"))
    if song_upload.value:
        _f = song_upload.value[0]
        _stem, _ext = _f.name.rsplit(".", 1)
        SONG = INPUT_DIR / f"{safe_name(_stem)}.{_ext.lower()}"
        SONG.write_bytes(_f.contents)
    else:
        mo.stop(not song_url.value.strip(), mo.callout("请上传文件或填写链接", kind="warn"))
        _rc, _tail = sh([VPY, TOOL, "fetch", "--url", song_url.value.strip(), "--out-dir", INPUT_DIR])
        _res = json.loads([l for l in _tail if l.startswith("RESULT_JSON=")][-1][12:])
        _p = Path(_res["path"])
        SONG = _p.with_name(safe_name(_p.stem) + _p.suffix)
        _p.rename(SONG)
    SONG_OUT = OUT_DIR / SONG.stem
    SONG_OUT.mkdir(parents=True, exist_ok=True)
    mo.vstack([mo.md(f"**已载入：** `{SONG.name}`"), mo.audio(str(SONG))])
    return SONG, SONG_OUT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## ④ 人声分离（UVR / Roformer）""")
    return


@app.cell(hide_code=True)
def _(mo):
    VOCAL_MODELS = {
        "MelBand Roformer Kim FT3 (unwa) · 推荐": "mel_band_roformer_kim_ft3_unwa.ckpt",
        "BS-Roformer Viperx 1297 · 经典高分": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "MelBand Roformer Kim (原版)": "vocals_mel_band_roformer.ckpt",
        "MelBand Roformer Vocals (becruily)": "mel_band_roformer_vocals_becruily.ckpt",
        "MDX23C InstVoc HQ": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
        "UVR-MDX-NET Voc FT (ONNX, 快)": "UVR-MDX-NET-Voc_FT.onnx",
    }
    KARAOKE_MODELS = {
        "不分离和声": "",
        "MelBand Karaoke (becruily) · 推荐": "mel_band_roformer_karaoke_becruily.ckpt",
        "Mel-Roformer Karaoke (aufr33 & viperx)": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "BS Roformer Karaoke (frazer & becruily)": "bs_roformer_karaoke_frazer_becruily.ckpt",
    }
    DEREVERB_MODELS = {
        "不去混响": "",
        "BS-Roformer De-Reverb · 推荐": "deverb_bs_roformer_8_384dim_10depth.ckpt",
        "MDX23C De-Reverb (aufr33 & jarredou)": "MDX23C-De-Reverb-aufr33-jarredou.ckpt",
        "UVR-DeEcho-DeReverb (VR)": "UVR-DeEcho-DeReverb.pth",
    }
    DENOISE_MODELS = {
        "不降噪": "",
        "Mel-Roformer Denoise (aufr33)": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        "Mel-Roformer Denoise Aggr (aufr33)": "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
    }
    sep_vocal = mo.ui.dropdown(VOCAL_MODELS, value="MelBand Roformer Kim FT3 (unwa) · 推荐", label="1. 人声/伴奏")
    sep_karaoke = mo.ui.dropdown(KARAOKE_MODELS, value="MelBand Karaoke (becruily) · 推荐", label="2. 主唱/和声")
    sep_dereverb = mo.ui.dropdown(DEREVERB_MODELS, value="BS-Roformer De-Reverb · 推荐", label="3. 去混响")
    sep_denoise = mo.ui.dropdown(DENOISE_MODELS, value="不降噪", label="4. 降噪")
    sep_skip = mo.ui.checkbox(label="输入已经是干声（跳过分离，直接转换）")
    sep_btn = mo.ui.run_button(label="✂️ 开始分离", kind="success")
    mo.vstack([mo.hstack([sep_vocal, sep_karaoke]), mo.hstack([sep_dereverb, sep_denoise]),
               sep_skip, sep_btn])
    return sep_btn, sep_denoise, sep_dereverb, sep_karaoke, sep_skip, sep_vocal


@app.cell
def _(
    SEP_MODELS, SONG, SONG_OUT, TOOL, VPY, json, mo, sep_btn, sep_denoise,
    sep_dereverb, sep_karaoke, sep_skip, sep_vocal, sh, shutil,
):
    mo.stop(not sep_btn.value, mo.md("_选好模型后点「开始分离」（模型首次使用会自动下载）_"))

    def _run_sep(inp, model, prefix):
        _rc, _tail = sh([VPY, TOOL, "separate", "--input", inp, "--model", model, "--out-dir", SONG_OUT,
                         "--prefix", prefix, "--model-dir", SEP_MODELS], quiet_re=r"\d+%\|")
        return json.loads([l for l in _tail if l.startswith("RESULT_JSON=")][-1][12:])

    STEMS = {}
    if sep_skip.value:
        _dry = SONG_OUT / "lead_dry.wav"
        sh(["ffmpeg", "-y", "-loglevel", "error", "-i", SONG, "-ar", "44100", _dry])
        STEMS["lead"] = str(_dry)
    else:
        _r = _run_sep(SONG, sep_vocal.value, "1_vocals")
        STEMS["vocals"], STEMS["instrumental"] = _r["main"], _r["rest"]
        _lead = _r["main"]
        if sep_karaoke.value:
            _r = _run_sep(_lead, sep_karaoke.value, "2_lead")
            _lead, STEMS["backing"] = _r["main"], _r["rest"]
        if sep_dereverb.value:
            _r = _run_sep(_lead, sep_dereverb.value, "3_dereverb")
            _lead = _r["main"]
        if sep_denoise.value:
            _r = _run_sep(_lead, sep_denoise.value, "4_denoise")
            _lead = _r["main"]
        _final = SONG_OUT / "lead_dry.wav"
        shutil.copy(_lead, _final)
        STEMS["lead"] = str(_final)

    _labels = {"lead": "🎯 主唱干声（将被转换）", "backing": "和声", "instrumental": "伴奏", "vocals": "完整人声"}
    mo.vstack([mo.md("### 分离结果")] +
              [mo.vstack([mo.md(f"**{_labels[k]}**"), mo.audio(v)]) for k, v in STEMS.items()])
    return (STEMS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## ⑤ RVC 音色转换""")
    return


@app.cell(hide_code=True)
def _(mo):
    rvc_pitch = mo.ui.slider(-24, 24, 1, value=0, label="变调（半音；男→女 +12，女→男 -12）", show_value=True, include_input=True, full_width=True)
    rvc_f0 = mo.ui.dropdown(["rmvpe", "fcpe", "crepe", "crepe-tiny", "hybrid[rmvpe+fcpe]", "hybrid[crepe+rmvpe]",
                             "hybrid[crepe+fcpe]", "hybrid[crepe+rmvpe+fcpe]"], value="rmvpe", label="音高提取算法")
    rvc_index = mo.ui.slider(0, 1, 0.05, value=0.5, label="特征检索比例 index rate（高=更像目标，低=更少杂音）", show_value=True, full_width=True)
    rvc_rms = mo.ui.slider(0, 1, 0.05, value=0.25, label="响度包络混合 volume envelope（1=用模型响度）", show_value=True, full_width=True)
    rvc_protect = mo.ui.slider(0, 0.5, 0.01, value=0.33, label="辅音/气声保护 protect（0.5=关闭）", show_value=True, full_width=True)
    rvc_embedder = mo.ui.dropdown(["contentvec", "spin", "spin-v2", "chinese-hubert-base", "japanese-hubert-base",
                                   "korean-hubert-base"], value="contentvec", label="嵌入模型（须与训练时一致，大多为 contentvec）")
    rvc_split = mo.ui.checkbox(label="分段推理（长歌省显存）")
    rvc_autotune = mo.ui.checkbox(label="自动修音 Autotune")
    rvc_autotune_s = mo.ui.slider(0, 1, 0.1, value=1.0, label="修音强度", show_value=True)
    rvc_clean = mo.ui.checkbox(label="降噪 clean audio")
    rvc_clean_s = mo.ui.slider(0, 1, 0.1, value=0.5, label="降噪强度", show_value=True)
    rvc_formant = mo.ui.checkbox(label="共振峰偏移 formant shifting")
    rvc_formant_q = mo.ui.slider(1, 16, 0.1, value=1.0, label="formant quefrency", show_value=True)
    rvc_formant_t = mo.ui.slider(1, 16, 0.1, value=1.0, label="formant timbre", show_value=True)
    rvc_backing = mo.ui.checkbox(label="和声也用同一模型转换", value=False)
    rvc_btn = mo.ui.run_button(label="🔁 开始转换", kind="success")
    mo.vstack([
        rvc_pitch, rvc_index, rvc_rms, rvc_protect,
        mo.hstack([rvc_f0, rvc_embedder], justify="start"),
        mo.accordion({"高级选项": mo.vstack([
            mo.hstack([rvc_split, rvc_backing], justify="start"),
            mo.hstack([rvc_autotune, rvc_autotune_s], justify="start"),
            mo.hstack([rvc_clean, rvc_clean_s], justify="start"),
            mo.hstack([rvc_formant, rvc_formant_q, rvc_formant_t], justify="start"),
        ])}),
        rvc_btn,
    ])
    return (
        rvc_autotune, rvc_autotune_s, rvc_backing, rvc_btn, rvc_clean,
        rvc_clean_s, rvc_embedder, rvc_f0, rvc_formant, rvc_formant_q,
        rvc_formant_t, rvc_index, rvc_pitch, rvc_protect, rvc_rms, rvc_split,
    )


@app.cell
def _(
    APPLIO, MODELS, SONG_OUT, STEMS, VPY, mo, rvc_autotune, rvc_autotune_s,
    rvc_backing, rvc_btn, rvc_clean, rvc_clean_s, rvc_embedder, rvc_f0,
    rvc_formant, rvc_formant_q, rvc_formant_t, rvc_index, rvc_pitch,
    rvc_protect, rvc_rms, rvc_split, sh, voice_model,
):
    mo.stop(not rvc_btn.value, mo.md("_调好参数后点「开始转换」_"))
    mo.stop(not voice_model.value, mo.callout("请先在第 ② 步选择声音模型", kind="danger"))
    _pth, _index = MODELS[voice_model.value]

    def _infer(inp, out):
        _cmd = [VPY, "core.py", "infer", "--input-path", inp, "--output-path", out,
                "--pth-path", _pth, "--index-path", _index or "",
                "--pitch", int(rvc_pitch.value), "--f0-method", rvc_f0.value,
                "--index-rate", rvc_index.value if _index else 0, "--volume-envelope", rvc_rms.value,
                "--protect", rvc_protect.value, "--embedder-model", rvc_embedder.value,
                "--export-format", "WAV", "--clean-strength", rvc_clean_s.value,
                "--f0-autotune-strength", rvc_autotune_s.value,
                "--formant-qfrency", rvc_formant_q.value, "--formant-timbre", rvc_formant_t.value]
        if rvc_split.value:
            _cmd.append("--split-audio")
        if rvc_autotune.value:
            _cmd.append("--f0-autotune")
        if rvc_clean.value:
            _cmd.append("--clean-audio")
        if rvc_formant.value:
            _cmd.append("--formant-shifting")
        sh(_cmd, cwd=APPLIO)

    _tag = f"{voice_model.value}_p{int(rvc_pitch.value)}"
    CONVERTED = {"lead": str(SONG_OUT / f"rvc_lead_{_tag}.wav")}
    _infer(STEMS["lead"], CONVERTED["lead"])
    if rvc_backing.value and "backing" in STEMS:
        CONVERTED["backing"] = str(SONG_OUT / f"rvc_backing_{_tag}.wav")
        _infer(STEMS["backing"], CONVERTED["backing"])
    mo.vstack([mo.md(f"### 转换完成（模型 **{voice_model.value}**）")] +
              [mo.vstack([mo.md(f"**{k}**"), mo.audio(v)]) for k, v in CONVERTED.items()])
    return (CONVERTED,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## ⑥ 混音导出""")
    return


@app.cell(hide_code=True)
def _(mo):
    mix_lead_db = mo.ui.slider(-12, 12, 0.5, value=0, label="主唱音量 dB", show_value=True, full_width=True)
    mix_inst_db = mo.ui.slider(-12, 12, 0.5, value=0, label="伴奏音量 dB", show_value=True, full_width=True)
    mix_back_db = mo.ui.slider(-24, 6, 0.5, value=-3, label="和声音量 dB", show_value=True, full_width=True)
    mix_reverb = mo.ui.slider(0, 0.6, 0.01, value=0.12, label="混响湿度", show_value=True, full_width=True)
    mix_room = mo.ui.slider(0, 1, 0.05, value=0.3, label="房间大小", show_value=True, full_width=True)
    mix_comp = mo.ui.checkbox(label="主唱压缩", value=True)
    mix_follow = mo.ui.checkbox(label="伴奏/和声跟随变调（变调不是 ±12 的整倍数时勾选）")
    mix_btn = mo.ui.run_button(label="🎚️ 混音并导出", kind="success")
    mo.vstack([mix_lead_db, mix_inst_db, mix_back_db, mix_reverb, mix_room,
               mo.hstack([mix_comp, mix_follow], justify="start"), mix_btn])
    return (
        mix_back_db, mix_btn, mix_comp, mix_follow, mix_inst_db, mix_lead_db,
        mix_reverb, mix_room,
    )


@app.cell
def _(
    CONVERTED, SONG, SONG_OUT, STEMS, TOOL, VPY, file_dl, json, mix_back_db,
    mix_btn, mix_comp, mix_follow, mix_inst_db, mix_lead_db, mix_reverb,
    mix_room, mo, rvc_pitch, sh, voice_model, zipfile,
):
    mo.stop(not mix_btn.value, mo.md("_点「混音并导出」生成成品_"))
    _pitch = int(rvc_pitch.value)
    _follow = _pitch if mix_follow.value else 0
    _out = SONG_OUT / f"{SONG.stem}_{voice_model.value}_cover.wav"
    _cmd = [VPY, TOOL, "mix", "--lead", CONVERTED["lead"], "--out", _out,
            "--lead-db", mix_lead_db.value, "--inst-db", mix_inst_db.value,
            "--backing-db", mix_back_db.value, "--reverb", mix_reverb.value,
            "--room", mix_room.value, "--inst-pitch", _follow]
    if "instrumental" in STEMS:
        _cmd += ["--inst", STEMS["instrumental"]]
    _bk = CONVERTED.get("backing") or STEMS.get("backing")
    if _bk:
        _cmd += ["--backing", _bk]
    if mix_comp.value:
        _cmd.append("--compress")
    _rc, _tail = sh(_cmd)
    _res = json.loads([l for l in _tail if l.startswith("RESULT_JSON=")][-1][12:])
    _final = _res.get("mp3") or _res["wav"]

    _zip = SONG_OUT / f"{SONG.stem}_{voice_model.value}_all_stems.zip"
    with zipfile.ZipFile(_zip, "w", zipfile.ZIP_DEFLATED) as _z:
        for _p in list(STEMS.values()) + list(CONVERTED.values()) + list(_res.values()):
            _z.write(_p, arcname=_p.split("/")[-1])
    mo.vstack([
        mo.md("## 🎉 成品"),
        mo.audio(_final),
        mo.hstack([file_dl(_final, "⬇️ 下载成品"), file_dl(_res["wav"], "⬇️ 无损 WAV"),
                   file_dl(_zip, "⬇️ 全部分轨 zip")], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## ⑦ 训练自己的声音模型（可选）

        数据：10–30 分钟干净干声（无伴奏、无混响），上传 `.zip` 或多个音频文件。
        在 96 GB 显存的 RTX Pro 6000 上，batch size 可以开大（16–32）。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    tr_name = mo.ui.text(label="模型名称（英文/数字）", value="MyVoice")
    tr_files = mo.ui.file(filetypes=[".zip", ".wav", ".mp3", ".flac", ".m4a", ".ogg"], multiple=True,
                          kind="area", max_size=3_000_000_000, label="上传数据集（zip 或多个音频）")
    tr_sr = mo.ui.dropdown(["32000", "40000", "48000"], value="40000", label="采样率")
    tr_f0 = mo.ui.dropdown(["rmvpe", "fcpe", "crepe", "crepe-tiny"], value="rmvpe", label="音高算法")
    tr_embedder = mo.ui.dropdown(["contentvec", "spin", "spin-v2", "chinese-hubert-base", "japanese-hubert-base",
                                  "korean-hubert-base"], value="contentvec", label="嵌入模型")
    tr_vocoder = mo.ui.dropdown(["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"], value="HiFi-GAN", label="声码器")
    tr_cut = mo.ui.dropdown(["Automatic", "Simple", "Skip"], value="Automatic", label="切片方式")
    tr_nr = mo.ui.checkbox(label="预处理降噪")
    tr_epochs = mo.ui.number(10, 2000, 10, value=300, label="总轮数 epochs")
    tr_batch = mo.ui.number(1, 50, 1, value=16, label="batch size")
    tr_save = mo.ui.number(1, 100, 1, value=25, label="每 N 轮保存")
    tr_cache = mo.ui.checkbox(label="数据缓存到显存（更快）", value=True)
    tr_cleanup = mo.ui.checkbox(label="清除同名旧训练重新开始")
    tr_btn = mo.ui.run_button(label="🏋️ 预处理 → 提取特征 → 训练 → 建索引", kind="success")
    mo.vstack([tr_name, tr_files,
               mo.hstack([tr_sr, tr_f0, tr_embedder], justify="start"),
               mo.hstack([tr_vocoder, tr_cut, tr_nr], justify="start"),
               mo.hstack([tr_epochs, tr_batch, tr_save], justify="start"),
               mo.hstack([tr_cache, tr_cleanup], justify="start"), tr_btn])
    return (
        tr_batch, tr_btn, tr_cache, tr_cleanup, tr_cut, tr_embedder, tr_epochs,
        tr_f0, tr_files, tr_name, tr_nr, tr_save, tr_sr, tr_vocoder,
    )


@app.cell
def _(
    APPLIO, DATASETS, EXPORTS, LOGS, VPY, file_dl, get_model_ver, io,
    is_installed, mo, os, re, safe_name, set_model_ver, sh, shutil, tr_batch,
    tr_btn, tr_cache, tr_cleanup, tr_cut, tr_embedder, tr_epochs, tr_f0,
    tr_files, tr_name, tr_nr, tr_save, tr_sr, tr_vocoder, zipfile,
):
    mo.stop(not tr_btn.value)
    mo.stop(not is_installed(), mo.callout("请先完成第 ① 步安装", kind="danger"))
    _name = safe_name(tr_name.value)
    _ds = DATASETS / _name
    if tr_files.value:
        if _ds.exists():
            shutil.rmtree(_ds)
        _ds.mkdir(parents=True)
        for _f in tr_files.value:
            if _f.name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(_f.contents)) as _z:
                    for _m in _z.namelist():
                        if _m.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")) and "__MACOSX" not in _m:
                            (_ds / safe_name(_m.replace("/", "_").rsplit(".", 1)[0])).with_suffix("." + _m.rsplit(".", 1)[1]).write_bytes(_z.read(_m))
            else:
                (_ds / _f.name).write_bytes(_f.contents)
    mo.stop(not _ds.exists() or not any(_ds.iterdir()), mo.callout("请上传数据集", kind="warn"))
    print(f"数据集文件数：{len(list(_ds.iterdir()))}")

    _cores = str(min(os.cpu_count() or 4, 64))
    if tr_cleanup.value and (LOGS / _name).exists():
        shutil.rmtree(LOGS / _name)
    _pp = [VPY, "core.py", "preprocess", "--model-name", _name, "--dataset-path", _ds, "--sample-rate", tr_sr.value,
           "--cpu-cores", _cores, "--cut-preprocess", tr_cut.value, "--chunk-len", "3.0", "--overlap-len", "0.3",
           "--normalization-mode", "none", "--noise-reduction-strength", "0.7"]
    if tr_nr.value:
        _pp.append("--noise-reduction")
    sh(_pp, cwd=APPLIO)
    sh([VPY, "core.py", "extract", "--model-name", _name, "--f0-method", tr_f0.value, "--sample-rate", tr_sr.value,
        "--cpu-cores", _cores, "--gpu", "0", "--embedder-model", tr_embedder.value, "--include-mutes", "2"], cwd=APPLIO)
    _tr = [VPY, "core.py", "train", "--model-name", _name, "--save-every-epoch", int(tr_save.value),
           "--total-epoch", int(tr_epochs.value), "--sample-rate", tr_sr.value, "--batch-size", int(tr_batch.value),
           "--gpu", "0", "--vocoder", tr_vocoder.value, "--pretrained", "--save-only-latest", "--save-every-weights"]
    if tr_cache.value:
        _tr.append("--cache-data-in-gpu")
    sh(_tr, cwd=APPLIO, quiet_re=r"^\s*\d+%\|")
    sh([VPY, "core.py", "index", "--model-name", _name, "--index-algorithm", "Auto"], cwd=APPLIO)

    # 导出推理包：最新权重 + index
    _d = LOGS / _name
    _w = sorted([p for p in _d.glob(f"{_name}_*e_*s.pth")], key=lambda p: p.stat().st_mtime)
    _idx = sorted(_d.glob("*.index"), key=lambda p: p.stat().st_mtime)
    mo.stop(not _w, mo.callout("未找到训练权重，请查看上方日志", kind="danger"))
    _zip = EXPORTS / f"{_name}_infer.zip"
    with zipfile.ZipFile(_zip, "w") as _z:
        _z.write(_w[-1], arcname=f"{_name}/{_w[-1].name}")
        if _idx:
            _z.write(_idx[-1], arcname=f"{_name}/{_idx[-1].name}")
    set_model_ver(get_model_ver() + 1)
    mo.vstack([mo.callout(mo.md(f"✅ 训练完成：`{_w[-1].name}`，已出现在第 ② 步模型列表"), kind="success"),
               file_dl(_zip, "⬇️ 下载模型（pth + index）")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## ⑧ Applio 完整 WebUI（可选）

        启动 Gradio 公网链接，获得 Applio 全部界面：推理、批量推理、训练、TTS、模型融合、插件、TensorBoard 等。
        （云端无声卡，「实时变声」页不可用。）
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    ui_start = mo.ui.run_button(label="🌐 启动 WebUI", kind="success")
    ui_stop = mo.ui.run_button(label="⏹ 停止 WebUI", kind="danger")
    mo.hstack([ui_start, ui_stop], justify="start")
    return ui_start, ui_stop


@app.cell
def _(APPLIO, ROOT, VPY, is_installed, mo, re, subprocess, time, ui_start, ui_stop, venv_env):
    import signal as _signal
    _pidf = ROOT / "webui.pid"
    _log = ROOT / "webui.log"
    if ui_stop.value and _pidf.exists():
        try:
            import os as _os
            _os.killpg(int(_pidf.read_text()), _signal.SIGTERM)
        except Exception as _e:
            print("停止时出错：", _e)
        _pidf.unlink(missing_ok=True)
        _msg = mo.callout("WebUI 已停止", kind="neutral")
    elif ui_start.value:
        mo.stop(not is_installed(), mo.callout("请先完成第 ① 步安装", kind="danger"))
        _p = subprocess.Popen([str(VPY), "app.py", "--share", "--server-name", "0.0.0.0", "--port", "6969"],
                              cwd=str(APPLIO), env=venv_env(), stdout=open(_log, "w"), stderr=subprocess.STDOUT,
                              start_new_session=True)
        _pidf.write_text(str(_p.pid))
        _url = None
        for _ in range(180):
            time.sleep(1)
            _txt = _log.read_text(errors="ignore")
            _m = re.search(r"https://[\w\-]+\.gradio\.live", _txt)
            if _m:
                _url = _m.group(0)
                break
            if _p.poll() is not None:
                break
        _tail = "\n".join(_log.read_text(errors="ignore").splitlines()[-25:])
        _msg = (mo.callout(mo.md(f"### ✅ WebUI 已启动：[{_url}]({_url})\n（72 小时有效，关闭笔记本即失效）"), kind="success")
                if _url else mo.vstack([mo.callout("未获取到公网链接，日志如下：", kind="warn"), mo.plain_text(_tail)]))
    else:
        _msg = mo.md("_点击启动后约 30–90 秒出现公网链接_")
    _msg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ### 💡 调参速查
        | 现象 | 调整 |
        |---|---|
        | 音色不像 | `index rate` ↑（0.6–0.8）；换更好的模型 |
        | 电音/破音 | 音高算法换 `rmvpe` / `hybrid[rmvpe+fcpe]`；`protect` ↑；输入先去混响 |
        | 齿音/气声糊 | `protect` 调到 0.4–0.5 |
        | 调太高/太低 | 变调 ±12；非 12 整数倍时勾选「伴奏跟随变调」 |
        | 人声太干/太湿 | 第 ⑥ 步混响湿度 |
        | 显存/时长问题 | 勾选「分段推理」 |
        """
    )
    return


if __name__ == "__main__":
    app.run()
