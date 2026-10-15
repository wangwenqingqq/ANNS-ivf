#!/usr/bin/env python3
"""Run the preregistered diagnostic, preserving failures and excluding busy GPUs."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apps(uuid):
    result = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    return [int(row[1]) for row in csv.reader(result.splitlines()) if row and row[0].strip() == uuid]


def belongs(pid, parent):
    for _ in range(64):
        if pid == parent:
            return True
        if pid <= 1:
            return False
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except FileNotFoundError:
            return False
    return False


def run_case(command, stem, out, env, uuid):
    if apps(uuid):
        raise RuntimeError("Selected GPU busy; no new process launched")
    metadata = {"command": command, "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out / "raw" / f"{stem}.start.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (out / "raw" / f"{stem}.csv").open("w") as stdout, (out / "raw" / f"{stem}.stderr").open("w") as stderr:
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        metadata["pid"] = process.pid
        start = time.monotonic()
        contaminated = False
        try:
            while process.poll() is None:
                time.sleep(1)
                foreign = [pid for pid in apps(uuid) if not belongs(pid, process.pid)]
                if foreign or time.monotonic() - start > 300:
                    contaminated = True
                    process.terminate()
                    break
            rc = process.wait(timeout=20)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)
            raise
    metadata.update(returncode=rc, contaminated_or_timeout=contaminated, elapsed_seconds=time.monotonic()-start)
    (out / "raw" / f"{stem}.status.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if rc or contaminated:
        raise RuntimeError(f"{stem} failed (exit={rc}, contaminated={contaminated}); evidence retained")
    print(f"PASS {stem}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        from unittest.mock import patch
        assert belongs(os.getpid(), os.getpid())
        assert not belongs(1, os.getpid())
        with patch.object(Path, "read_text", side_effect=FileNotFoundError):
            assert not belongs(123456789, os.getpid())
        print("process-isolation regression checks: PASS")
        return
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    lock = open(f"/tmp/anns-churn-{uuid}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if apps(uuid):
        raise RuntimeError("Selected GPU already has a compute process")
    out = (args.out or HERE / "runs" / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "source", "build"):
        (out / name).mkdir()
    for name in ("probe.cu", "build.sh", "run.py", "CONTRACT.md", "PRIOR_ART.md"):
        shutil.copyfile(HERE / name, out / "source" / name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    environment = {
        "parent_commit": subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip(),
        "gpu": subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=name,uuid,driver_version,compute_cap,power.limit,clocks.sm,clocks.mem", "--format=csv"], text=True),
        "cuda": subprocess.check_output([os.environ.get("CUDACXX", "/usr/local/cuda/bin/nvcc"), "--version"], text=True),
        "mode": "pilot" if args.pilot else "full", "scope": "diagnostic GPU serialized epoch; no concurrent service claim",
    }
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    with (out / "build" / "build.log").open("w") as log:
        subprocess.run([env.get("CUDACXX", "/usr/local/cuda/bin/nvcc"), "-O3", "-std=c++20", "-arch=sm_120", "-lineinfo", "-Xptxas=-v", str(out / "source/probe.cu"), "-o", str(out / "build/probe")], check=True, stdout=log, stderr=subprocess.STDOUT)
    binary = str(out / "build/probe")
    try:
        for dim in (128, 768):
            for tool in ("memcheck", "synccheck"):
                run_case(["compute-sanitizer", "--tool", tool, "--error-exitcode", "99", binary,
                          str(dim), "32", "random", "4", "PFI", "small"], f"gate_{tool}_{dim}", out, env, uuid)
        if not args.pilot:
            for workload in ("fifo", "random"):
                run_case([binary, "128", "128", workload, "1000", "PFI"], f"stress_{workload}", out, env, uuid)
        for process, order in enumerate(("PFI",) if args.pilot else ("PFI", "IFP", "PFI", "IFP")):
            for dim in (128,) if args.pilot else (128, 768):
                for batch in (128,) if args.pilot else (128, 1024):
                    for workload in ("fifo", "random", "hot"):
                        run_case([binary, str(dim), str(batch), workload, "8" if args.pilot else "80", order],
                                 f"p{process}_d{dim}_b{batch}_{workload}", out, env, uuid)
        (out / "COMPLETED.json").write_text(json.dumps({"complete": True, "mode": environment["mode"],
            "binary_sha256": digest(out / "build/probe"), "source_sha256": digest(out / "source/probe.cu")}, indent=2) + "\n")
    finally:
        manifest = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": digest(p)}
                    for p in sorted(out.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"]
        (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Retained: {out}")


if __name__ == "__main__":
    main()
