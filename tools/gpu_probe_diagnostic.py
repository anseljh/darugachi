#!/usr/bin/env python3
"""Report which GPU-status signals actually work on this box.

Darugachi wants to answer "is the GPU too busy to accept work right now?"
NVIDIA's own known-limitations list for CUDA on WSL 2 names GPU utilization
and active compute processes among the NVML queries that are not supported
there, so the signals a status endpoint would rely on cannot be assumed --
they have to be measured on the actual box, with the actual driver.

This script measures them. It probes, in order:

  1. the WSL nvidia-smi shim (/usr/lib/wsl/lib/nvidia-smi via PATH)
  2. the Windows nvidia-smi.exe through /mnt/c interop
  3. per-process compute apps (expected to be unavailable under WDDM)
  4. a repeated-sample utilization window, to see whether the number moves
  5. nvidia-smi dmon
  6. Windows "GPU Engine" performance counters through powershell.exe,
     which is the only route that attributes GPU use to a process
  7. llama-swap's own HTTP API, to find which path reports running models

Nothing here changes any state; every probe is a read with a timeout.

Run it twice and compare:

    tools/gpu_probe_diagnostic.py --label idle --out /tmp/gpu-idle.json
    # start a game, let it get past the menu into actual rendering
    tools/gpu_probe_diagnostic.py --label game --out /tmp/gpu-game.json
    diff <(jq -S . /tmp/gpu-idle.json) <(jq -S . /tmp/gpu-game.json)

A field that reports plausibly in both runs AND differs between them is one
a status endpoint can use. A field that reads the same idle and busy is
worse than useless -- it would make the endpoint confidently wrong.
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

# Every field a /gpu endpoint might plausibly want, queried in one call so the
# report shows which of them survive this driver and this WSL version.
QUERY_FIELDS = [
    "name",
    "driver_version",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "enforced.power.limit",
    "clocks.sm",
    "clocks.max.sm",
    "pstate",
    "temperature.gpu",
    "fan.speed",
    "clocks_throttle_reasons.active",
]

# Fields sampled repeatedly in probe 4. Numeric, and the ones that would carry
# a "substantially busy" decision.
SAMPLE_FIELDS = ["utilization.gpu", "utilization.memory", "power.draw", "clocks.sm", "memory.used"]

LINUX_SMI_CANDIDATES = ["/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi"]
WINDOWS_SMI_CANDIDATES = [
    "/mnt/c/Windows/System32/nvidia-smi.exe",
    "/mnt/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe",
]
POWERSHELL_CANDIDATES = ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"]

# Paths that different llama-swap versions have used to report running models.
# GET only -- nothing here may start or stop a model.
LLAMA_SWAP_PATHS = ["/running", "/api/models", "/api/models/running", "/v1/models", "/health"]

SMI_TIMEOUT = 10
DMON_TIMEOUT = 20
POWERSHELL_TIMEOUT = 30
HTTP_TIMEOUT = 5

# A value nvidia-smi prints when it cannot answer. WSL's shim and WDDM both
# produce several spellings of this, so match loosely rather than exactly.
UNSUPPORTED_MARKERS = ("n/a", "not supported", "not available", "unknown", "insufficient permissions")

INSTANCE = re.compile(r"pid_(?P<pid>\d+).*?engtype_(?P<engtype>[A-Za-z0-9]+)")


def find_binary(candidates: list[str], name: str | None = None) -> str | None:
    if name:
        found = shutil.which(name)
        if found:
            return found
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def run(command: list[str], timeout: int) -> dict:
    """Run a command and report what happened, never raising."""
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "seconds": round(time.monotonic() - started, 3),
            "outcome": "timeout",
            "timeout": timeout,
        }
    except OSError as error:
        return {
            "command": command,
            "seconds": round(time.monotonic() - started, 3),
            "outcome": "oserror",
            "error": str(error),
        }
    return {
        "command": command,
        "seconds": round(time.monotonic() - started, 3),
        "outcome": "ran",
        "returncode": completed.returncode,
        # Windows binaries reached through interop emit CRLF.
        "stdout": completed.stdout.replace("\r\n", "\n").strip(),
        "stderr": completed.stderr.replace("\r\n", "\n").strip(),
    }


def classify(value: str) -> str:
    """Say whether one queried field came back usable."""
    text = value.strip().strip("[]").strip()
    lowered = text.lower()
    if not text:
        return "empty"
    if "err" in lowered:
        return "error"
    if any(marker in lowered for marker in UNSUPPORTED_MARKERS):
        return "unsupported"
    return "ok"


def query_fields(binary: str, fields: list[str]) -> dict:
    command = [
        binary,
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    result = run(command, SMI_TIMEOUT)
    if result["outcome"] != "ran" or result["returncode"] != 0:
        return result
    rows = [row for row in result["stdout"].splitlines() if row.strip()]
    result["gpu_count"] = len(rows)
    parsed = []
    for row in rows:
        values = [value.strip() for value in row.split(",")]
        # A short row means the driver rejected a field name outright; pad so the
        # report still lines names up with what did come back.
        values += [""] * (len(fields) - len(values))
        parsed.append(
            {field: {"value": value, "verdict": classify(value)} for field, value in zip(fields, values)}
        )
    result["fields"] = parsed
    return result


def sample_utilization(binary: str, samples: int, interval: float) -> dict:
    """Query the moving fields repeatedly.

    utilization.gpu is the fraction of a sample period in which any kernel ran,
    and that period is between 1 second and 1/6 second depending on the card, so
    a single reading is noise. This also times each call, which is the latency
    budget a status endpoint would inherit.
    """
    readings = []
    latencies = []
    for index in range(samples):
        if index:
            time.sleep(interval)
        result = query_fields(binary, SAMPLE_FIELDS)
        latencies.append(result["seconds"])
        if "fields" not in result:
            readings.append({"error": result})
            continue
        readings.append({field: data["value"] for field, data in result["fields"][0].items()})

    summary = {}
    for field in SAMPLE_FIELDS:
        values = []
        for reading in readings:
            raw = reading.get(field, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        if values:
            summary[field] = {
                "numeric_samples": len(values),
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
                "all_identical": len(set(values)) == 1,
            }
        else:
            summary[field] = {"numeric_samples": 0, "note": "no numeric readings"}
    return {
        "samples": samples,
        "interval_seconds": interval,
        "readings": readings,
        "summary": summary,
        "call_latency_seconds": {
            "min": round(min(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "max": round(max(latencies), 3),
        },
    }


def probe_compute_apps(binary: str) -> dict:
    return run(
        [
            binary,
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        SMI_TIMEOUT,
    )


def probe_dmon(binary: str, samples: int) -> dict:
    # -s pucm asks for power, utilization, clocks and memory columns.
    return run([binary, "dmon", "-s", "pucm", "-c", str(samples)], DMON_TIMEOUT)


def probe_perf_counters(powershell: str) -> dict:
    """Read Windows GPU Engine counters -- what Task Manager's graphs show.

    This is the only probe that attributes GPU work to a process, and the only
    one that sees a game's graphics engines rather than compute kernels.
    """
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' "
        "-SampleInterval 1 -MaxSamples 1).CounterSamples | "
        "Where-Object { $_.CookedValue -gt 0.5 } | "
        "Select-Object InstanceName, CookedValue | "
        "ConvertTo-Csv -NoTypeInformation"
    )
    result = run([powershell, "-NoProfile", "-NonInteractive", "-Command", script], POWERSHELL_TIMEOUT)
    if result["outcome"] != "ran" or result["returncode"] != 0:
        return result

    by_engine_type: dict[str, float] = {}
    by_pid: dict[str, float] = {}
    rows = []
    for line in result["stdout"].splitlines()[1:]:  # first line is the CSV header
        parts = [part.strip().strip('"') for part in line.split('","')]
        if len(parts) < 2:
            continue
        instance = parts[0].strip('"')
        try:
            value = float(parts[1].strip('"'))
        except ValueError:
            continue
        rows.append({"instance": instance, "percent": round(value, 2)})
        match = INSTANCE.search(instance)
        if not match:
            continue
        engine_type = match.group("engtype").lower()
        pid = match.group("pid")
        # Instances are per-engine, so these sums can exceed 100 -- they measure
        # how many engines are busy, not a share of the whole card.
        by_engine_type[engine_type] = round(by_engine_type.get(engine_type, 0.0) + value, 2)
        by_pid[pid] = round(by_pid.get(pid, 0.0) + value, 2)
    result["active_instances"] = sorted(rows, key=lambda row: -row["percent"])
    result["percent_by_engine_type"] = by_engine_type
    result["percent_by_pid"] = dict(sorted(by_pid.items(), key=lambda item: -item[1]))
    return result


def probe_llama_swap(port: str, api_key: str) -> dict:
    findings = {}
    for path in LLAMA_SWAP_PATHS:
        url = f"http://127.0.0.1:{port}{path}"
        request = urllib.request.Request(url)
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                body = response.read(4096).decode("utf-8", "replace")
                findings[path] = {
                    "status": response.status,
                    "seconds": round(time.monotonic() - started, 3),
                    "body_head": body[:600],
                }
        except urllib.error.HTTPError as error:
            findings[path] = {"status": error.code, "reason": error.reason}
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            findings[path] = {"error": str(error)}
    return {"port": port, "authorized": bool(api_key), "paths": findings}


def usable_fields(probe: dict | None) -> set[str]:
    if not probe or "fields" not in probe or not probe["fields"]:
        return set()
    return {field for field, data in probe["fields"][0].items() if data["verdict"] == "ok"}


def summarize(report: dict) -> list[str]:
    """Say what the numbers mean for the endpoint's design."""
    notes = []
    linux = report["probes"].get("linux_smi_query")
    windows = report["probes"].get("windows_smi_query")
    counters = report["probes"].get("windows_perf_counters")

    linux_ok = usable_fields(linux)
    windows_ok = usable_fields(windows)

    if not linux_ok and not windows_ok:
        notes.append("No nvidia-smi query worked at all -- check the driver before reading anything else here.")
    else:
        for field in ("memory.free", "memory.used", "memory.total"):
            if field in linux_ok:
                notes.append(f"{field}: usable from the WSL shim.")
            elif field in windows_ok:
                notes.append(f"{field}: only usable via nvidia-smi.exe interop.")
            else:
                notes.append(f"{field}: UNAVAILABLE from both binaries -- the VRAM gate cannot be built on it.")

    for field in ("utilization.gpu", "power.draw", "clocks.sm", "pstate"):
        if field in linux_ok:
            notes.append(f"{field}: usable from the WSL shim.")
        elif field in windows_ok:
            notes.append(f"{field}: unavailable in WSL, usable via nvidia-smi.exe interop.")
        else:
            notes.append(f"{field}: unavailable from both nvidia-smi binaries.")

    window = report["probes"].get("utilization_window", {}).get("summary", {})
    for field, data in window.items():
        # Only the utilization fields are flagged for never moving. memory.used
        # and clocks.sm holding steady across a few seconds is ordinary; a
        # utilization figure pinned to one value is the documented WSL failure.
        if not field.startswith("utilization."):
            continue
        if data.get("numeric_samples", 0) > 1 and data.get("all_identical"):
            notes.append(
                f"{field} never moved across {data['numeric_samples']} samples (constant {data['min']}) "
                "-- suspect a stub value rather than a real reading, and confirm against the busy run."
            )

    latency = report["probes"].get("utilization_window", {}).get("call_latency_seconds")
    if latency:
        notes.append(
            f"One nvidia-smi query costs {latency['median']}s at the median, {latency['max']}s worst -- "
            "cache the result rather than probing per request."
        )

    if counters and counters.get("outcome") == "ran" and counters.get("returncode") == 0:
        engines = counters.get("percent_by_engine_type") or {}
        notes.append(
            f"GPU Engine perf counters work ({counters['seconds']}s per call); "
            f"active engine types: {engines or 'none right now'}."
        )
        if "3d" in engines:
            notes.append("engtype_3d is active -- this is the signal that catches a game.")
    elif counters:
        notes.append("GPU Engine perf counters failed -- per-process attribution is not available on this box.")

    swap = report["probes"].get("llama_swap", {}).get("paths", {})
    working = [path for path, data in swap.items() if data.get("status") == 200]
    notes.append(
        f"llama-swap answered 200 on: {working}" if working
        else "llama-swap answered 200 on nothing -- check it is running and whether these paths need auth."
    )
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="unlabeled", help="tag for this run, e.g. idle or game")
    parser.add_argument("--samples", type=int, default=5, help="utilization samples to take (default 5)")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between samples (default 1.0)")
    parser.add_argument("--out", help="also write the full report as JSON to this path")
    parser.add_argument("--json", action="store_true", help="print JSON only, no human-readable report")
    arguments = parser.parse_args()

    linux_smi = find_binary(LINUX_SMI_CANDIDATES, "nvidia-smi")
    windows_smi = find_binary(WINDOWS_SMI_CANDIDATES, "nvidia-smi.exe")
    powershell = find_binary(POWERSHELL_CANDIDATES, "powershell.exe")

    report = {
        "label": arguments.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "binaries": {"linux_smi": linux_smi, "windows_smi": windows_smi, "powershell": powershell},
        "probes": {},
    }
    probes = report["probes"]

    if linux_smi:
        probes["linux_smi_query"] = query_fields(linux_smi, QUERY_FIELDS)
        probes["linux_smi_compute_apps"] = probe_compute_apps(linux_smi)
        probes["linux_smi_dmon"] = probe_dmon(linux_smi, min(arguments.samples, 5))
        probes["utilization_window"] = sample_utilization(linux_smi, arguments.samples, arguments.interval)
    if windows_smi:
        probes["windows_smi_query"] = query_fields(windows_smi, QUERY_FIELDS)
        probes["windows_smi_compute_apps"] = probe_compute_apps(windows_smi)
    if powershell:
        probes["windows_perf_counters"] = probe_perf_counters(powershell)
    probes["llama_swap"] = probe_llama_swap(
        os.environ.get("PORT", "9292"), os.environ.get("API_KEY", "")
    )

    report["notes"] = summarize(report)

    if arguments.out:
        Path(arguments.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if arguments.json:
        print(json.dumps(report, indent=2))
        return

    print(f"=== GPU probe diagnostic ({report['label']}) {report['timestamp']}")
    for name, path in report["binaries"].items():
        print(f"  {name}: {path or 'NOT FOUND'}")

    for name in ("linux_smi_query", "windows_smi_query"):
        probe = probes.get(name)
        if not probe:
            continue
        print(f"\n--- {name} ({probe.get('seconds')}s, outcome {probe['outcome']}, rc {probe.get('returncode')})")
        if probe.get("stderr"):
            print(f"    stderr: {probe['stderr']}")
        for index, fields in enumerate(probe.get("fields", [])):
            if probe.get("gpu_count", 1) > 1:
                print(f"    GPU {index}:")
            for field, data in fields.items():
                marker = " " if data["verdict"] == "ok" else "!"
                print(f"    {marker} {field:34} {data['verdict']:12} {data['value']}")

    window = probes.get("utilization_window")
    if window:
        print(f"\n--- utilization_window ({window['samples']} samples @ {window['interval_seconds']}s)")
        print(f"    call latency: {window['call_latency_seconds']}")
        for field, data in window["summary"].items():
            print(f"    {field:22} {data}")

    for name in ("linux_smi_compute_apps", "windows_smi_compute_apps", "linux_smi_dmon"):
        probe = probes.get(name)
        if not probe:
            continue
        print(f"\n--- {name} ({probe.get('seconds')}s, outcome {probe['outcome']}, rc {probe.get('returncode')})")
        for stream in ("stdout", "stderr"):
            if probe.get(stream):
                for line in probe[stream].splitlines()[:12]:
                    print(f"    {stream}: {line}")

    counters = probes.get("windows_perf_counters")
    if counters:
        print(f"\n--- windows_perf_counters ({counters.get('seconds')}s, outcome {counters['outcome']}, rc {counters.get('returncode')})")
        if counters.get("stderr"):
            print(f"    stderr: {counters['stderr'].splitlines()[0]}")
        if counters.get("percent_by_engine_type") is not None:
            print(f"    by engine type: {counters['percent_by_engine_type']}")
            print(f"    by pid:         {counters['percent_by_pid']}")
            for row in counters["active_instances"][:10]:
                print(f"    {row['percent']:8.2f}  {row['instance']}")

    swap = probes.get("llama_swap", {})
    print(f"\n--- llama_swap (port {swap.get('port')}, bearer sent: {swap.get('authorized')})")
    for path, data in swap.get("paths", {}).items():
        print(f"    {path:24} {data}")

    print("\n=== Notes")
    for note in report["notes"]:
        print(f"  - {note}")
    print(
        "\nRun this once idle and once with a game rendering, then diff the JSON. "
        "A field that reads the same in both cannot carry the busy/idle decision."
    )


if __name__ == "__main__":
    main()
