"""A stdio tap for MCP (extension 1.2.0, T42): starts the real server as a child process, relays every line in both
directions unchanged and appends it to a JSONL log, so the lesson can show the messages a client and a server exchange.

    python labs/M14/mcp_tap.py LOG.jsonl -- python labs/M14/mcp_server.py --session s1
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time


def relay(source, sink, log, direction, lock):
    for line in iter(source.readline, b""):
        sink.write(line)
        sink.flush()
        with lock:
            log.write(json.dumps({"t": round(time.time(), 3), "dir": direction, "line": line.decode("utf-8", "replace").rstrip("\n")},
                                 ensure_ascii=False) + "\n")
            log.flush()
    if direction == "client->server":
        sink.close()  # the client closed stdin: let the server see end of input and exit


def main(argv):
    split = argv.index("--")
    log_path, command = argv[split - 1], argv[split + 1:]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
    lock = threading.Lock()
    with open(log_path, "a", encoding="utf-8", newline="\n") as log:
        threads = [threading.Thread(target=relay, args=(sys.stdin.buffer, child.stdin, log, "client->server", lock), daemon=True),
                   threading.Thread(target=relay, args=(child.stdout, sys.stdout.buffer, log, "server->client", lock), daemon=True)]
        for thread in threads:
            thread.start()
        code = child.wait()
        threads[1].join(timeout=5)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
