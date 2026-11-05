"""Throwaway local PostgreSQL 18 cluster with pgvector for M10-L08X (extension 1.2.0, T34).

Nothing is written into the installed PostgreSQL: pgvector is built from the pinned official source into
artifacts/M10/pgvector, and a separate cluster in artifacts/M10/pgdata listens only on 127.0.0.1:55433.
PostgreSQL 18 finds the extension there through extension_control_path and dynamic_library_path.
The cluster trusts loopback connections without a password: synthetic course data only, never real data.

    python labs/M10/pg_local.py build    # once: download pgvector 0.8.7 source, compile with MSVC (Windows)
    python labs/M10/pg_local.py init     # once: create the cluster
    python labs/M10/pg_local.py start | stop | status
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PG_ROOT = Path(os.environ.get("M10_PG_ROOT", r"C:\Program Files\PostgreSQL\18"))
PGVECTOR = {"version": "0.8.7", "url": "https://github.com/pgvector/pgvector/archive/refs/tags/v0.8.7.zip",
            "sha256": "7b3b8188a5cc0aa96db3c8f3fca1e78a8a612c46545065490c76448705c6b29e"}
WORK = ROOT / "artifacts/M10/pgvector"
EXTENSION = WORK / "ext"  # share/extension/*.control, *.sql and lib/vector.dll
DATA = ROOT / "artifacts/M10/pgdata"
PORT = 55433
USER = "course"
CONNINFO = f"host=127.0.0.1 port={PORT} user={USER} dbname=postgres"


def tool(name: str) -> str:
    path = PG_ROOT / "bin" / f"{name}.exe"
    return str(path) if path.exists() else shutil.which(name) or name


def vcvars() -> Path:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    found = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                            "-property", "installationPath"], capture_output=True, text=True, check=False).stdout.strip()
    if not found:
        raise SystemExit("Visual Studio C++ build tools not found: install the 'Desktop development with C++' workload")
    return Path(found) / "VC/Auxiliary/Build/vcvars64.bat"


def build() -> Path:
    archive = WORK / "download" / f"pgvector-{PGVECTOR['version']}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        urllib.request.urlretrieve(PGVECTOR["url"], archive.with_suffix(".partial"))
        archive.with_suffix(".partial").replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != PGVECTOR["sha256"]:
        raise SystemExit(f"sha256 mismatch for {archive.name}: {digest}")
    source = WORK / "src"
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            if not (source / member.filename).resolve().is_relative_to(source.resolve()):
                raise ValueError("Unsafe ZIP member")
        package.extractall(source)
    folder = source / f"pgvector-{PGVECTOR['version']}"
    # 'all' only: the 'install' target would copy into the PostgreSQL installation under Program Files.
    script = WORK / "build.cmd"  # a file, not a cmd /c string: avoids nested quoting of paths with spaces
    script.write_text(f'@echo off\r\ncall "{vcvars()}" >nul || exit /b 1\r\ncd /d "{folder}" || exit /b 1\r\n'
                      f'nmake /nologo /F Makefile.win PGROOT="{PG_ROOT}" all\r\n', encoding="ascii")
    subprocess.run(["cmd", "/d", "/c", str(script)], check=True)
    (EXTENSION / "share/extension").mkdir(parents=True, exist_ok=True)
    (EXTENSION / "lib").mkdir(parents=True, exist_ok=True)
    shutil.copy2(folder / "vector.dll", EXTENSION / "lib")
    shutil.copy2(folder / "vector.control", EXTENSION / "share/extension")
    for script in (folder / "sql").glob("vector--*.sql"):
        shutil.copy2(script, EXTENSION / "share/extension")
    print(f"pgvector {PGVECTOR['version']} built into {EXTENSION}")
    return EXTENSION


def init() -> Path:
    version = subprocess.run([tool("postgres"), "--version"], capture_output=True, text=True, check=True).stdout
    if int(version.split()[-1].split(".")[0]) < 18:
        raise SystemExit(f"{version.strip()}: extension_control_path needs PostgreSQL 18; with 17 install pgvector into the server instead")
    if not (EXTENSION / "lib/vector.dll").exists():
        raise SystemExit("Run 'pg_local.py build' first")
    if not (DATA / "PG_VERSION").exists():
        subprocess.run([tool("initdb"), "-D", str(DATA), "-U", USER, "-A", "trust", "-E", "UTF8", "--no-locale"], check=True)
        settings = [f"port = {PORT}", "listen_addresses = '127.0.0.1'",
                    f"extension_control_path = '{(EXTENSION / 'share').as_posix()};$system'",
                    f"dynamic_library_path = '{(EXTENSION / 'lib').as_posix()};$libdir'"]
        with (DATA / "postgresql.conf").open("a", encoding="utf-8") as conf:
            conf.write("\n# M10-L08X throwaway cluster\n" + "\n".join(settings) + "\n")
    print(f"cluster ready: {DATA}")
    return DATA


def pg_ctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([tool("pg_ctl"), "-D", str(DATA), *args], capture_output=True, text=True, check=False)


def running() -> bool:
    return (DATA / "PG_VERSION").exists() and pg_ctl("status").returncode == 0


def start() -> None:
    if not running():
        # postgres is started directly in its own process group with a hidden console: started through pg_ctl from
        # a terminal, new backends fail with 0xC0000142 once that terminal closes. Logs go to a file, not to pipes.
        log = (DATA.parent / "pgdata.log").open("ab")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.Popen([tool("postgres"), "-D", str(DATA)], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         creationflags=flags, start_new_session=os.name != "nt")
        for _ in range(60):
            if subprocess.run([tool("pg_isready"), "-h", "127.0.0.1", "-p", str(PORT)], capture_output=True, check=False).returncode == 0:
                break
            time.sleep(0.5)
        else:
            raise SystemExit(f"server did not start, see {DATA.parent / 'pgdata.log'}")
    print(f"running: {CONNINFO}")


def stop() -> None:
    if running():
        pg_ctl("-m", "fast", "-w", "stop")
    print("stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "init", "start", "stop", "status"])
    command = parser.parse_args().command
    if command == "status":
        print(f"running: {CONNINFO}" if running() else "stopped")
    else:
        {"build": build, "init": init, "start": start, "stop": stop}[command]()
