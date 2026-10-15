#!/usr/bin/env python3
"""Collect native DEBR component evidence, not an ANN throughput benchmark."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
# Reuse the already-tested device-isolation helpers, including fail-closed ancestry.
SPEC = importlib.util.spec_from_file_location("churn_runner", HERE.parent / "churn-frontier-20261015/run.py")
ISOLATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ISOLATION)
PIN = "f755006d78ccca565f97cda053727d4d48b82d46"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_case(command, stem, out, env, uuid):
    if ISOLATION.apps(uuid):
        raise RuntimeError("GPU occupied; no process launched")
    status = {"command": command, "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out / "raw" / (stem + ".start.json")).write_text(json.dumps(status, indent=2) + "\n")
    with (out / "raw" / (stem + ".jsonl")).open("w") as stdout, (out / "raw" / (stem + ".stderr")).open("w") as stderr:
        child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        status["pid"] = child.pid
        start = time.monotonic()
        contaminated = False
        try:
            while child.poll() is None:
                time.sleep(.5)
                if child.poll() is not None:
                    break
                foreign = [pid for pid in ISOLATION.apps(uuid) if not ISOLATION.belongs(pid, child.pid)]
                if foreign or time.monotonic() - start > 120:
                    contaminated = True
                    status["foreign_pids"] = foreign
                    status["foreign_proc_status"] = {}
                    for pid in foreign:
                        try:
                            status["foreign_proc_status"][str(pid)] = Path(f"/proc/{pid}/status").read_text()
                        except OSError as error:
                            status["foreign_proc_status"][str(pid)] = str(error)
                    child.terminate()
                    break
            code = child.wait(timeout=20)
        except BaseException:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=20)
            raise
    status.update(returncode=code, contaminated_or_timeout=contaminated, elapsed_wall_seconds=time.monotonic()-start)
    (out / "raw" / (stem + ".status.json")).write_text(json.dumps(status, indent=2) + "\n")
    if code or contaminated:
        raise RuntimeError(f"{stem} failed; raw failure preserved")
    print("PASS", stem, flush=True)


def self_test():
    import tempfile
    from unittest.mock import patch
    assert ISOLATION.belongs(os.getpid(), os.getpid())
    with patch.object(Path, "read_text", side_effect=FileNotFoundError):
        assert not ISOLATION.belongs(123456789, os.getpid())
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(ISOLATION, "apps", return_value=[7]), patch.object(subprocess, "Popen") as launch:
            try:
                run_case(["never-launch"], "guard", Path(directory), {}, "mock-device")
            except RuntimeError as error:
                assert "occupied" in str(error)
            else:
                raise AssertionError("busy device accepted")
            launch.assert_not_called()
            assert not list(Path(directory).iterdir())
    print("Busy-device/no-launch and ancestry tests: PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=6)
    p.add_argument("--out", type=Path)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    lock = open(f"/tmp/anns-churn-{uuid}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if ISOLATION.apps(uuid):
        raise RuntimeError("GPU occupied")
    out = (args.out or HERE / "runs" / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "source", "build"):
        (out / name).mkdir()
    for name in ("probe.cu", "build.sh", "run.py", "CONTRACT.md", "PRIOR_ART.md"):
        shutil.copyfile(HERE/name, out/"source"/name)
    external = HERE / "external/MVGpuBTree"
    assert subprocess.check_output(["git", "-C", str(external), "rev-parse", "HEAD"], text=True).strip() == PIN
    assert not subprocess.check_output(["git", "-C", str(external), "status", "--porcelain"], text=True).strip()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    environment = {
        "parent_commit": subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip(),
        "gpu": subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=name,uuid,driver_version,compute_cap,power.limit,clocks.sm,clocks.mem", "--format=csv"], text=True),
        "cuda": subprocess.check_output([env.get("CUDACXX", "/usr/local/cuda/bin/nvcc"), "--version"], text=True),
        "gcc": subprocess.check_output(["g++", "--version"], text=True),
        "upstream_commit": PIN, "mode": "pilot" if args.pilot else "full",
        "upstream_header_hashes": {str(p.relative_to(external)): digest(p) for p in sorted((external/"include").glob("*.hpp"))},
    }
    (out/"environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    try:
        with (out/"build/build.log").open("w") as log:
            subprocess.run([env.get("CUDACXX", "/usr/local/cuda/bin/nvcc"), "-O3", "-std=c++20", "-arch=sm_120", "-lineinfo", "--expt-relaxed-constexpr", "-Xptxas=-v", "-I"+str(external/"include"), str(out/"source/probe.cu"), "-o", str(out/"build/probe")], stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = str(out/"build/probe")
        run_case([binary,"128","64","0","500","EL","1","64","0"], "smoke", out, env, uuid)
        for dim in (128,768):
            for tool in ("memcheck","synccheck","initcheck","racecheck"):
                run_case(["compute-sanitizer","--tool",tool,"--error-exitcode","99",binary,str(dim),"64","500","500","EL","1","8" if tool == "racecheck" else "64","0"], f"gate_{tool}_{dim}",out,env,uuid)
        if not args.pilot:
            run_case([binary,"128","64","0","2000","LE","1","8192","0"], "stress",out,env,uuid)
            for process,order in enumerate(("EL","LE","EL","LE")):
                for dim in (128,768):
                    for slots in (64,256):
                        for scenario,pre,hold in (("none",0,0),("pre",2000,0),("inside",0,2000)):
                            run_case([binary,str(dim),str(slots),str(pre),str(hold),order,"4","512","1"], f"p{process}_d{dim}_s{slots}_{scenario}",out,env,uuid)
        (out/"COMPLETED.json").write_text(json.dumps({"complete":True,"mode":environment["mode"],"source_sha256":digest(out/"source/probe.cu"),"binary_sha256":digest(out/"build/probe"),"native_header_sha256":digest(external/"include/memory_reclaimer.hpp")},indent=2)+"\n")
    finally:
        manifest=[{"path":str(p.relative_to(out)),"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(out.rglob("*")) if p.is_file() and p.name!="MANIFEST.json"]
        (out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__":
    main()
