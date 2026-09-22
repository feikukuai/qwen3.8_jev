#!/usr/bin/env python3
"""
One-click deploy for System One / JEV-style decisions on CPU.

Downloads llama.cpp (prebuilt if possible, else builds it), fetches the model,
starts llama-server with the RIGHT flags, and runs a smoke decision.

The flags matter more than the code:
  --parallel 1   a single large slot. --parallel N shards the context and makes
                 the shared-prefix cache thrash (measured: alternating 1.6s/58s)
  -c 8192        enough context for a warm prefix plus branch suffixes
  -t <cores>     physical cores, not hyperthreads, is usually fastest
  -ngl 0         CPU only

Usage:
    python3 deploy.py                      # defaults: 27b tier
    python3 deploy.py --tier 4b            # small tier
    python3 deploy.py --tier 27b-gsq       # ISTA-DASLab GSQ-RCO IQ3_S
    python3 deploy.py --list
"""
from __future__ import annotations
import argparse, json, os, platform, shutil, subprocess, sys, time, urllib.request

MODELS = {
    "27b": {
        "name": "Qwen3.5-27B-Q3_K_S",
        "repo": "unsloth/Qwen3.5-27B-GGUF",
        "file": "Qwen3.5-27B-Q3_K_S.gguf",
        "size_gb": 12.29,
        "note": "Standard Q3_K_S. Best speed/quality balance for 16 GB+ RAM.",
    },
    "27b-gsq": {
        "name": "Qwen3.8-27B-GSQ-RCO-IQ3_S",
        "repo": "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF",
        "file": "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
        "size_gb": 11.77,
        "note": "GSQ+RCO mixed-precision ~3.06 bpw. Highest quality at ~12 GB.",
    },
    "4b": {
        "name": "Qwen3.5-4B-Q4_K_M",
        "repo": "unsloth/Qwen3.5-4B-GGUF",
        "file": "Qwen3.5-4B-Q4_K_M.gguf",
        "size_gb": 2.74,
        "note": "Small tier. Runs on a laptop; use this if RAM is tight.",
    },
}

HF = "https://huggingface.co/{repo}/resolve/main/{file}"
MIRROR = "https://hf-mirror.com/{repo}/resolve/main/{file}"   # for users behind a slow link


def log(msg):
    print("[deploy] " + msg, flush=True)


def have(cmd):
    return shutil.which(cmd) is not None


def cpu_threads():
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 4
    return max(1, n)


def download(url, dest, connections=16):
    log("downloading %s" % url)
    if have("aria2c"):
        cmd = ["aria2c", "-x", str(connections), "-s", str(connections), "-k", "8M",
               "--continue=true", "--file-allocation=none",
               "--console-log-level=warn", "-o", os.path.basename(dest),
               "-d", os.path.dirname(dest) or ".", url]
    else:
        cmd = ["curl", "-L", "-C", "-", "--retry", "5", "-o", dest, url]
    subprocess.check_call(cmd)


def ensure_model(tier, model_dir, mirror=False):
    spec = MODELS[tier]
    dest = os.path.join(model_dir, spec["file"])
    if os.path.exists(dest) and os.path.getsize(dest) > spec["size_gb"] * 1e9 * 0.97:
        log("model already present: %s" % dest)
        return dest
    tpl = MIRROR if mirror else HF
    urls = [tpl.format(repo=spec["repo"], file=spec["file"])]
    if not mirror:
        urls.append(MIRROR.format(repo=spec["repo"], file=spec["file"]))
    last = None
    for u in urls:
        try:
            download(u, dest)
            if os.path.getsize(dest) > spec["size_gb"] * 1e9 * 0.97:
                return dest
        except Exception as e:
            last = e
            log("download failed from %s (%s), trying next" % (u.split('/')[2], e))
    raise SystemExit("could not download model: %s" % last)


def ensure_llama(workdir):
    """Return path to llama-server, building llama.cpp if needed."""
    if os.environ.get("LLAMA_SERVER"):
        return os.environ["LLAMA_SERVER"]
    root = os.path.join(workdir, "llama.cpp")
    server = os.path.join(root, "build", "bin", "llama-server")
    if os.path.exists(server):
        return server
    if not os.path.exists(root):
        log("cloning llama.cpp ...")
        subprocess.check_call(["git", "clone", "--depth", "1",
                               "https://github.com/ggml-org/llama.cpp.git", root])
    log("configuring (native CPU + OpenBLAS if available) ...")
    cfg = ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
           "-DGGML_NATIVE=ON", "-DGGML_CCACHE=OFF"]
    # OpenBLAS helps prefill a lot on AMD/Intel; skip silently if absent
    try:
        subprocess.check_call(["pkg-config", "--exists", "openblas"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cfg += ["-DGGML_BLAS=ON", "-DGGML_BLAS_VENDOR=OpenBLAS"]
        log("OpenBLAS found - enabling BLAS backend")
    except Exception:
        log("OpenBLAS not found - building without BLAS (slower prefill; "
            "install libopenblas-dev for ~2x)")
    subprocess.check_call(cfg, cwd=root)
    subprocess.check_call(["cmake", "--build", "build", "--config", "Release",
                           "-j", str(cpu_threads())], cwd=root)
    if not os.path.exists(server):
        raise SystemExit("build finished but %s is missing" % server)
    return server


def serve(server, model, port, ctx, threads):
    cmd = [server, "-m", model, "--host", "127.0.0.1", "--port", str(port),
           "-t", str(threads), "-c", str(ctx), "-b", "1024", "-ub", "1024",
           "--parallel", "1", "-ngl", "0", "--no-warmup"]
    log("starting server: " + " ".join(cmd))
    return subprocess.Popen(cmd)


def wait_health(port, timeout=300):
    url = "http://127.0.0.1:%d/health" % port
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(2)
    return False


def smoke(port, model_label):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from system_one import SystemOne
    c = SystemOne("http://127.0.0.1:%d" % port, "local")
    review = ("The plot is thin, but the two leads are so charming that I left "
              "the cinema smiling.")
    t0 = time.time()
    r = c.decide(review, "What is the sentiment of this review?",
                 ["negative", "positive"])
    dt = time.time() - t0
    log("smoke decision: %s (p=%.3f) in %.2fs" % (r["choice"], r["confidence"], dt))
    ok = r["choice"] == "positive"
    log("smoke %s" % ("PASSED" if ok else "FAILED - unexpected answer"))
    if dt > 20:
        log("NOTE: first call rebuilds the hybrid recurrent state and is slow "
            "(~50s on a 27B). Later calls sharing the same prefix are ~1.5s.")
    return ok


def main():
    ap = argparse.ArgumentParser(description="One-click System One / JEV deploy")
    ap.add_argument("--tier", default="27b", choices=sorted(MODELS))
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "./models"))
    ap.add_argument("--workdir", default=os.environ.get("DEPLOY_DIR", "."))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mirror", action="store_true",
                    help="use hf-mirror.com first (slow international links)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in sorted(MODELS.items()):
            print("%-9s %-30s %5.2f GB  %s" % (k, v["name"], v["size_gb"], v["note"]))
        return

    if not have("cmake") or not have("git"):
        log("WARNING: cmake and git are required to build llama.cpp. "
            "Install them, or set LLAMA_SERVER=/path/to/llama-server.")

    os.makedirs(args.model_dir, exist_ok=True)
    threads = args.threads or cpu_threads()
    log("platform=%s cores=%d tier=%s" % (platform.platform(), threads, args.tier))

    model = ensure_model(args.tier, args.model_dir, mirror=args.mirror)
    server = ensure_llama(args.workdir)
    if args.no_serve:
        log("done (--no-serve); model=%s server=%s" % (model, server))
        return

    proc = serve(server, model, args.port, args.ctx, threads)
    try:
        if not wait_health(args.port):
            log("server did not become healthy")
            return 1
        log("server healthy at http://127.0.0.1:%d" % args.port)
        smoke(args.port, MODELS[args.tier]["name"])
        log("ready. Press Ctrl-C to stop.")
        proc.wait()
    except KeyboardInterrupt:
        log("stopping")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
