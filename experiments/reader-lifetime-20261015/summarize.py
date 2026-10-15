#!/usr/bin/env python3
"""Fail-closed native-DEBR evidence reader; pilot admission is never a full run."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def need(value, reason):
    if not value:
        raise ValueError(reason)


def read(root, admission=False):
    complete = json.loads((root/"COMPLETED.json").read_text())
    need(complete["complete"] and complete["mode"] in ("pilot", "full") and (admission or complete["mode"] == "full"), "complete full matrix required")
    manifest = json.loads((root/"MANIFEST.json").read_text())
    listed = [row["path"] for row in manifest]
    need(len(set(listed)) == len(listed), "duplicate manifest path")
    required = {"COMPLETED.json", "environment.json", "build/probe", "build/build.log"}
    required.update("source/"+n for n in ("probe.cu","run.py","build.sh","CONTRACT.md","PRIOR_ART.md"))
    stems = ["smoke"]+[f"gate_{tool}_{d}" for d in (128,768) for tool in ("memcheck","synccheck","initcheck")]
    if not admission:
        stems += [f"gate_racecheck_{d}" for d in (128,768)] + ["stress"]
        stems += [f"p{p}_d{d}_s{s}_{scene}" for p in range(4) for d in (128,768) for s in (64,256) for scene in ("none","pre","inside")]
    required.update(f"raw/{stem}.{ext}" for stem in stems for ext in ("jsonl","stderr","start.json","status.json"))
    need(required <= set(listed), "required evidence missing from manifest")
    for row in manifest:
        name = Path(row["path"])
        need(not name.is_absolute() and ".." not in name.parts, "unsafe path")
        need((root/name).resolve().is_relative_to(root.resolve()), "manifest escaped run root")
        data = (root/name).read_bytes()
        need(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], "artifact hash mismatch")
    for field, path in (("source_sha256","source/probe.cu"),("binary_sha256","build/probe")):
        need(complete[field] == hashlib.sha256((root/path).read_bytes()).hexdigest(), "unbound source/binary digest")
    upstream = json.loads((HERE/"UPSTREAM.json").read_text())
    env = json.loads((root/"environment.json").read_text())
    need(env["upstream_commit"] == upstream["commit"] and env["upstream_header_hashes"] == upstream["headers"], "upstream mismatch")
    need(complete["native_header_sha256"] == upstream["headers"]["include/memory_reclaimer.hpp"], "native digest mismatch")
    for stem in stems:
        status = json.loads((root/"raw"/(stem+".status.json")).read_text())
        need(status["returncode"] == 0 and not status["contaminated_or_timeout"], "failed or contaminated process")
        text = (root/"raw"/(stem+".jsonl")).read_text()+(root/"raw"/(stem+".stderr")).read_text()
        need("PASS: native DEBR" in text,"missing process oracle")
        if "gate_" in stem:
            if "racecheck" in stem:
                need(re.search(r"RACECHECK SUMMARY: 0 hazards",text),"racecheck not clean")
            else:
                need(re.findall(r"ERROR SUMMARY: (\d+) errors",text) == ["0"],"sanitizer not clean")
    return complete, stems


def analyze(root):
    complete, stems = read(root)
    observations=[]
    for process,order in enumerate(("EL","LE","EL","LE")):
        for dim in (128,768):
            for slots in (64,256):
                for scene,pre,hold in (("none",0,0),("pre",2000,0),("inside",0,2000)):
                    stem=f"p{process}_d{dim}_s{slots}_{scene}"
                    rows=[json.loads(line) for line in (root/"raw"/(stem+".jsonl")).read_text().splitlines()]
                    need(len(rows)==10,"incomplete trial matrix")
                    need([(r["scope"],r["observation"]) for r in rows]==[(scope,t) for scope in order for t in range(-1,4)],"order/trial drift")
                    for r in rows:
                        need((r["dim"],r["slots"],r["pre_us"],r["hold_us"],r["attempts"])==(dim,slots,pre,hold,512),"shape drift")
                        steps=r["steps"]
                        need(r["oracle"] and r["final_live"]==16 and r["ack"]+r["failed"]==512 and r["freed"]==r["ack"],"resource/oracle failure")
                        need(len(steps)==512 and sum(s[0] for s in steps)==r["ack"] and max(s[1] for s in steps)==r["peak_slots"]<=slots,"trace/ack drift")
                        failures=[i for i,s in enumerate(steps) if not s[0]]
                        need(r["first_fail"]==(failures[0] if failures else -1),"first-failure mismatch")
                        need(len(r["readers"])==8 and all(v[6]>=v[5] for v in r["readers"]),"reader interval mismatch")
                        if r["observation"]>=0:
                            observations.append({"process":process,"order":order,"scenario":scene,**{k:v for k,v in r.items() if k!="steps"}})
    return {"state":"full component diagnostic; no ANN performance promotion","source_sha256":complete["source_sha256"],"binary_sha256":complete["binary_sha256"],"upstream_commit":json.loads((HERE/"UPSTREAM.json").read_text())["commit"],"observations":observations}


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        try:read(root)
        except FileNotFoundError:pass
        else:raise AssertionError("empty evidence accepted")
        (root/"COMPLETED.json").write_text('{"complete":true,"mode":"pilot"}')
        try:read(root)
        except ValueError as e:assert "full matrix" in str(e)
        else:raise AssertionError("pilot accepted as full")
        (root/"COMPLETED.json").write_text('{"complete":true,"mode":"full"}')
        (root/"MANIFEST.json").write_text('[]')
        try:read(root)
        except ValueError as e:assert "required evidence" in str(e)
        else:raise AssertionError("empty manifest accepted")
        (root/"MANIFEST.json").write_text('[{"path":"x"},{"path":"x"}]')
        try:read(root)
        except ValueError as e:assert "duplicate" in str(e)
        else:raise AssertionError("duplicate accepted")
    print("Native DEBR evidence parser regression tests: PASS")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path,nargs="?")
    p.add_argument("--admission",action="store_true")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:self_test()
    if args.run:
        if args.admission:
            c,stems=read(args.run,True)
            output={"state":"initial admission only; racecheck/full matrix unvalidated",**c,"passed_cases":stems}
        else:output=analyze(args.run)
        print(json.dumps(output,indent=2))
