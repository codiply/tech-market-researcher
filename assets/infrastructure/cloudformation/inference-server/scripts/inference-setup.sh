#!/bin/bash
set -euo pipefail

# cloud-init runs this in a minimal environment where HOME is not
# set; default to root's home (the script runs as root).
export HOME="${HOME:-/root}"

# Template parameters, written by the user-data trampoline (see
# inference-server.yaml) before this script runs.
source /var/lib/inference-setup/params.env

# Every step is idempotent (cloud-init user-data runs once; if the
# fallback reboot below ever fires, the same script must be safe to
# re-run and pick up where it left off).
# High-level progress log: one timestamped line per key event, written
# as we go so this single file shows exactly where setup is (first line
# = start, last line = the terminal event). It is appended, never truncated, so a
# reboot re-run keeps the original start timestamp and the first/last
# timestamps still bracket the total wall-clock time.
LOG_FILE=/var/log/inference-setup.log
log() {
local ts
ts="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "$ts [inference-setup] $*" | tee -a "$LOG_FILE"
}
log "Setup started"

# 0. Dashboard: a tiny, dependency-free HTTP server on port 8888
# with two pages: /dashboard (main page: one-line server status from
# the setup log, live CPU/memory/load/GPU stats and sparkline history)
# and /logs (tabbed, self-refreshing logs); the root URL redirects to
# /dashboard. Installed first so the page is live as soon as Tailscale
# login gives the instance its tailnet name. Enabled (not just started)
# so it survives reboots. The logs contain only setup/infrastructure
# output; the dashboard shows the GPU once the NVIDIA driver is
# installed (it displays a notice until then).
cat > /usr/local/bin/inference-dashboard.py <<'EOF'
#!/usr/bin/env python3
"""Serve the inference server dashboard: a logs page and a live metrics page."""
import html
import http.server
import json
import subprocess
from urllib.parse import parse_qs, urlparse

PORT = 8888
DEFAULT_LINES = 500

# (tab key, tab label, file path or None, journal unit or None)
SOURCES = [
    ("setup", "Setup", "/var/log/inference-setup.log", None),
    ("cloud-init-output", "Cloud-init output", "/var/log/cloud-init-output.log", None),
    ("cloud-init", "Cloud-init", "/var/log/cloud-init.log", None),
    ("setup-reboot", "Setup re-run (journal)", None, "inference-setup"),
    ("vllm", "vLLM server (journal)", None, "vllm-serve"),
    ("boot-script", "Boot script", "/var/lib/cloud/instance/scripts/part-001", None),
]

def fetch(key, lines=DEFAULT_LINES):
    """Return the last `lines` lines of the source (or all if None), or None if unknown."""
    path = next((s[2] for s in SOURCES if s[0] == key), None)
    unit = next((s[3] for s in SOURCES if s[0] == key), None)
    if unit is not None:
        cmd = ["journalctl", "-u", unit, "--no-pager"]
        if lines is not None:
            cmd += ["-n", str(lines)]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True).stdout
        except FileNotFoundError:
            return "(journalctl not found)"
        return out or "(no entries yet)"
    if path is not None:
        try:
            with open(path, errors="replace") as f:
                content = f.read().splitlines()
        except FileNotFoundError:
            return f"(file not found: {path})"
        return "\n".join(content[-lines:] if lines is not None else content) or "(empty)"
    return None

def shell():
    buttons = "".join(
        f'<button data-tab="{k}">{html.escape(n)}</button>'
        for k, n, _, _ in SOURCES)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>inference server logs</title>
<style>
body{{background:#111;color:#ddd;font-family:monospace;margin:0;
height:100vh;display:flex;flex-direction:column;overflow:hidden}}
h2{{padding:.5em 1em;margin:0}}
nav{{padding:.5em 1em;background:#181818;border-bottom:1px solid #333}}
nav button[data-tab]{{background:#222;color:#ddd;border:1px solid #444;border-radius:4px;
margin-right:.5em;padding:.4em .8em;cursor:pointer;font-family:inherit}}
nav button[data-tab].active{{background:#2d6cdf;border-color:#2d6cdf;color:#fff}}
#follow{{display:none;margin-left:auto;background:#2d6cdf;color:#fff;
border:1px solid #2d6cdf;border-radius:4px;padding:.4em .8em;
cursor:pointer;font-family:inherit}}
select{{background:#222;color:#ddd;border:1px solid #444;border-radius:4px;
padding:.4em;font-family:inherit}}
pre{{flex:1;overflow-y:auto;padding:1em;white-space:pre-wrap}}
a.metlink{{background:#222;color:#9ecbff;border:1px solid #444;border-radius:4px;
padding:.4em .8em;margin-right:.5em;text-decoration:none;font-family:inherit}}
</style></head>
<body>
<h2>inference server logs (auto-refreshes every 5 s, <span id="status"></span>)</h2>
<nav><a class="metlink" href="/dashboard">Dashboard</a>{buttons}
<label>Lines: <select id="lines">
<option value="500">500</option>
<option value="2000">2000</option>
<option value="10000">10000</option>
<option value="all">All</option>
</select></label>
<button id="follow">&#9660; Follow latest</button>
</nav>
<pre id="content">&nbsp;</pre>
<script>
var tab = null;
var content = document.getElementById('content');
var linesSel = document.getElementById('lines');
var followBtn = document.getElementById('follow');
var following = true;
function atBottom() {{
  return content.scrollHeight - content.scrollTop - content.clientHeight < 40;
}}
function setFollowUI() {{
  followBtn.style.display = following ? 'none' : 'inline-block';
}}
function jumpToBottom() {{
  following = true;
  content.scrollTop = content.scrollHeight;
  setFollowUI();
}}
function show(name) {{
  tab = name;
  document.querySelectorAll('nav button[data-tab]').forEach(function(b) {{
    b.classList.toggle('active', b.dataset.tab === name);
  }});
  following = true;
  setFollowUI();
  refresh();
}}
function refresh() {{
  var label = linesSel.value === 'all' ? 'all lines' : linesSel.value + ' lines';
  document.getElementById('status').textContent = label;
  fetch('/log/' + tab + '?lines=' + encodeURIComponent(linesSel.value))
    .then(function(r) {{ return r.text(); }}).then(function(t) {{
      content.textContent = t;
      if (following) {{
        content.scrollTop = content.scrollHeight;
      }}
    }});
}}
content.addEventListener('scroll', function() {{
  // Pause auto-follow when the user scrolls up; resume when they
  // scroll back near the bottom (or hit the follow button).
  following = atBottom();
  setFollowUI();
}});
followBtn.onclick = jumpToBottom;
document.querySelectorAll('nav button[data-tab]').forEach(function(b) {{
  b.onclick = function() {{ show(b.dataset.tab); }};
}});
linesSel.onchange = function() {{
  localStorage.setItem('lines', linesSel.value);
  refresh();
}};
var saved = localStorage.getItem('lines');
if (saved && [...linesSel.options].some(function(o) {{ return o.value === saved; }})) {{
  linesSel.value = saved;
}}
setInterval(function() {{ if (tab) refresh(); }}, 5000);
show('setup');
</script>
</body></html>
"""

# --- Metrics --------------------------------------------------------

_last_cpu = None

def cpu_util_pct():
    """CPU utilization % since the previous call (None on the first call)."""
    global _last_cpu
    with open("/proc/stat") as f:
        cur = list(map(int, f.readline().split()[1:]))
    total, idle = sum(cur), cur[3] + (cur[4] if len(cur) > 4 else 0)
    util = None
    if _last_cpu is not None and total > _last_cpu[0]:
        dt, di = total - _last_cpu[0], idle - _last_cpu[1]
        util = round(100.0 * (dt - di) / dt, 1)
    _last_cpu = (total, idle)
    return util

def mem_gb():
    """(used, total) RAM in GB from /proc/meminfo."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k] = int(v.strip().split()[0])  # kB
    total, avail = info.get("MemTotal", 0), info.get("MemAvailable", 0)
    return (total - avail) / 1048576.0, total / 1048576.0

def load_avg():
    with open("/proc/loadavg") as f:
        return f.read().split()[:3]

def gpu_stats():
    """Per-GPU stats from nvidia-smi, or None until the driver is installed."""
    fields = ",".join(("name", "utilization.gpu", "memory.used", "memory.total",
                       "temperature.gpu", "power.draw", "clocks.sm"))
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    gpus = []
    for row in (r.strip() for r in out.stdout.strip().splitlines()):
        c = row.split(",")
        gpus.append({"name": c[0], "util": float(c[1]), "mem_used": int(c[2]),
                     "mem_total": int(c[3]), "temp": float(c[4]),
                     "power": float(c[5]), "clock": float(c[6])})
    return gpus

def setup_status():
    """Last non-empty line of the high-level setup log (one-line server status)."""
    try:
        with open("/var/log/inference-setup.log") as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines[-1] if lines else ""
    except FileNotFoundError:
        return ""

def dashboard_data():
    mem_used, mem_total = mem_gb()
    return {"status": setup_status(),
            "cpu_pct": cpu_util_pct(), "gpus": gpu_stats(),
            "mem": {"used": round(mem_used, 1), "total": round(mem_total, 1)},
            "load": load_avg()}

DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>inference server dashboard</title>
<style>
body{background:#111;color:#ddd;font-family:monospace;margin:0}
h2{padding:.5em 1em;margin:0}
nav{padding:.5em 1em;background:#181818;border-bottom:1px solid #333}
a{background:#222;color:#9ecbff;border:1px solid #444;border-radius:4px;
padding:.4em .8em;text-decoration:none;font-family:inherit}
section{padding:1em}
h3{margin:0 0 .5em;color:#9ecbff}
.row{display:flex;align-items:center;gap:1em;margin:.4em 0}
.bar{width:300px;height:14px;background:#222;border-radius:3px}
.bar div{height:100%;background:#2d6cdf;border-radius:3px}
.warn{color:#e0a35a}
.chrow{display:flex;align-items:center;gap:1em;margin:.3em 0}
.chrow .cl{width:240px}
.chrow svg{background:#181818;border-radius:3px}
.chrow .cv{width:100px;text-align:right;color:#9ecbff}
</style></head>
<body>
<h2>inference server dashboard (auto-refreshes every 2 s)</h2>
<nav><a href="/logs">Logs</a></nav>
<section><h3>Status</h3><div id="status">&hellip;</div></section>
<section id="gpu"><h3>GPU</h3><div class="warn">loading&hellip;</div></section>
<section id="sys"><h3>CPU / Memory</h3></section>
<section><h3>History (up to 15 minutes)</h3><div id="charts"></div></section>
<script>
// History is in memory only: each page load starts from an empty
// window (sample shape {t, c: cpu_pct, m: ram GB, g: gpus}, max 15 min at 2 s).
var M = 450, H = [];
function push(d) {
  H.push({t: Date.now(), c: d.cpu_pct, m: d.mem.used, g: d.gpus});
  if (H.length > M) H = H.slice(H.length - M);
}
function series(get) { return H.map(get); }
function chart(label, values, unit) {
  var w = 560, h = 48;
  var v = values.filter(function(x) { return x !== null; });
  if (!v.length || values.length < 2) return '';
  var min = Math.min.apply(null, v), max = Math.max.apply(null, v);
  if (max - min < 1e-6) max = min + 1;
  var pts = [];
  values.forEach(function(x, i) {
    if (x !== null) pts.push([i / (values.length - 1) * w, h - (x - min) / (max - min) * h]);
  });
  var line = pts.map(function(p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join(' ');
  var area = '0,' + h + ' ' + line + ' ' + pts[pts.length - 1][0].toFixed(1) + ',' + h;
  return '<div class="chrow"><span class="cl">' + label + '</span>' +
    '<svg width="' + w + '" height="' + h + '">' +
    '<polygon points="' + area + '" fill="#1d3d6b"/>' +
    '<polyline points="' + line + '" fill="none" stroke="#5b9bd5" stroke-width="1.5"/></svg>' +
    '<span class="cv">' + v[v.length - 1] + unit + '</span></div>';
}
function bar(p) { return '<div class="bar"><div style="width:' + p + '%"></div></div>' + p + '%'; }
function render(d) {
  document.getElementById('status').textContent = d.status || '(no status yet)';
  var g = document.getElementById('gpu');
  if (!d.gpus) {
    g.innerHTML = '<div class="warn">NVIDIA driver not available (yet) &mdash; setup may still be installing it</div>';
  } else {
    g.innerHTML = d.gpus.map(function(x, i) {
      return '<div class="row">GPU ' + i + ' (' + x.name + ') ' + bar(x.util) +
        ' &nbsp; VRAM ' + x.mem_used + '/' + x.mem_total + ' MiB &nbsp; '
        + x.temp + '&deg;C &nbsp; ' + x.power + ' W &nbsp; ' + x.clock + ' MHz</div>';
    }).join('');
  }
  var c = document.getElementById('sys');
  c.innerHTML = '<h3>CPU / Memory</h3>' +
    '<div class="row">CPU ' + bar(d.cpu_pct === null ? 0 : d.cpu_pct) + (d.cpu_pct === null ? ' (waiting for first sample)' : '') + '</div>' +
    '<div class="row">RAM ' + d.mem.used + ' / ' + d.mem.total + ' GB</div>' +
    '<div class="row">Load ' + d.load.join(' ') + '</div>';
  var charts = '';
  charts += chart('CPU', series(function(s) { return s.c; }), '%');
  charts += chart('RAM', series(function(s) { return s.m; }), ' GB');
  if (d.gpus) d.gpus.forEach(function(g, i) {
    charts += chart('GPU ' + i + ' utilization', series(function(s) { return s.g ? s.g[i].util : null; }), '%');
    charts += chart('GPU ' + i + ' VRAM', series(function(s) { return s.g ? s.g[i].mem_used : null; }), ' MiB');
    charts += chart('GPU ' + i + ' temperature', series(function(s) { return s.g ? s.g[i].temp : null; }), ' °C');
    charts += chart('GPU ' + i + ' power', series(function(s) { return s.g ? s.g[i].power : null; }), ' W');
  });
  document.getElementById('charts').innerHTML = charts || '<div class="warn">collecting samples&hellip;</div>';
}
function tick() { fetch('/dashboard/data').then(function(r) { return r.json(); }).then(function(d) { push(d); render(d); }); }
tick();
setInterval(tick, 2000);
</script>
</body></html>
"""

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(302)
            self.send_header("Location", "/dashboard")
            self.end_headers()
            return
        elif parsed.path == "/logs":
            body, ctype = shell().encode(), "text/html; charset=utf-8"
        elif parsed.path == "/dashboard":
            body, ctype = DASHBOARD_HTML.encode(), "text/html; charset=utf-8"
        elif parsed.path == "/dashboard/data":
            body, ctype = json.dumps(dashboard_data()).encode(), "application/json"
        elif parsed.path.startswith("/log/"):
            lines = DEFAULT_LINES
            for v in parse_qs(parsed.query).get("lines", []):
                if v == "all":
                    lines = None
                elif v.isdigit():
                    lines = int(v)
            content = fetch(parsed.path[len("/log/"):], lines)
            if content is None:
                self.send_response(404)
                self.end_headers()
                return
            body, ctype = content.encode(), "text/plain; charset=utf-8"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep requests out of the journal
        pass

http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
EOF
cat > /etc/systemd/system/inference-dashboard.service <<'EOF'
[Unit]
Description=Serve inference dashboard (logs and metrics) over HTTP
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /usr/local/bin/inference-dashboard.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable inference-dashboard.service
systemctl start inference-dashboard.service
log "Dashboard running on port 8888 (http://${TS_HOSTNAME}.<tailnet>.ts.net:8888)"

# 1. Tailscale
log "Installing Tailscale"
curl -fsSL https://tailscale.com/install.sh | sh

# Authenticate with the auth key from Parameter Store (SecureString),
# fetched with the AWS CLI (not installed on the AMI by default).
# The key is only ever held in a shell variable: it is never echoed
# or logged (cloud-init only records this script's stdout/stderr),
# and is cleared from the shell's memory as soon as login finishes.
log "Installing AWS CLI (to read the Tailscale auth key from Parameter Store)"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y unzip
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install --update
rm -rf /tmp/aws /tmp/awscliv2.zip
TS_AUTH_KEY="$(aws ssm get-parameter --name "${TS_AUTH_KEY_PARAM}" --with-decryption --query 'Parameter.Value' --output text)"
if [ -z "$TS_AUTH_KEY" ] || [ "$TS_AUTH_KEY" = "None" ]; then
log "ERROR: could not read Tailscale auth key from Parameter Store (${TS_AUTH_KEY_PARAM}); aborting"
exit 1
fi
tailscale login --advertise-tags=tag:${TS_TAG} --hostname="${TS_HOSTNAME}" --operator=ubuntu --auth-key="$TS_AUTH_KEY"
unset TS_AUTH_KEY
log "Tailscale authenticated"
# 2. NVIDIA driver (meta-package: kernel module + firmware + utils)
# plus nvtop for live GPU monitoring.
log "Installing NVIDIA driver"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-580-server nvtop
# On the very first boot the box has no NVIDIA driver yet, so udev
# auto-loads nouveau, which claims the GPU. The blacklist the driver
# package installs only takes effect from the *next* boot, so unload
# nouveau here (a headless box has no open references); if it is in
# use, the reboot fallback below recovers, because the blacklist then
# applies and nouveau does not re-claim the GPU.
if lsmod | grep -q '^nouveau'; then
modprobe -r nouveau || true
fi
# DKMS builds the kernel module for the *running* kernel and runs
# depmod, so the module can normally be loaded immediately with
# modprobe.
modprobe nvidia nvidia_uvm nvidia-modeset || true
# Check that the driver actually owns a GPU, not merely that the module
# is loaded: the module can load with zero devices (e.g. if another
# driver still claims the GPU) and still pass a bare lsmod check, which
# would let a broken box continue into the vLLM install and fail there.
if ! nvidia-smi -L >/dev/null 2>&1; then
log "NVIDIA driver has no GPU (module not bound); rebooting"
# cloud-init runs user-data only once, so a reboot would leave the
# remaining steps unrun. Install a oneshot service that re-runs
# this same script after the reboot; a successful run removes it.
cat > /etc/systemd/system/inference-setup.service <<'EOF'
[Unit]
Description=Re-run inference server setup after reboot
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
# The re-run can block for many minutes (it re-runs the whole script,
# including the slow vLLM install). Disable the oneshot start timeout
# so systemd does not SIGTERM the script part-way through; the script
# self-terminates on its own. Use infinity, not 0, to reliably disable.
TimeoutStartSec=infinity
ExecStart=/bin/bash /var/lib/cloud/instance/scripts/part-001
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable inference-setup.service
log "REBOOT: restarting now to load the NVIDIA kernel module; setup resumes via the inference-setup service after reboot"
reboot
# reboot is asynchronous and does not stop this script; stop here so
# we do not keep installing into a box that is shutting down.
exit 0
fi

log "NVIDIA driver ready"

# 3. uv (installer places it in $HOME/.local/bin)
log "Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# 4. vLLM in an isolated Python 3.12 env at /opt/vllm
log "Setting up vLLM environment"
mkdir -p /opt/vllm
cd /opt/vllm
# Shared, world-readable HF cache so models downloaded as root are
# visible to the ubuntu user and reused across launches. Owned by
# ubuntu so it can also download models (root bypasses perms).
export HF_HOME=/opt/vllm/hf-cache
mkdir -p "$HF_HOME"
chown ubuntu:ubuntu "$HF_HOME"
# Install uv's managed Python under /opt/vllm instead of /root:
# /root is mode 700, so the ubuntu user cannot execute the venv's
# python (a symlink into /root) and every vllm launch fails with
# "bad interpreter: Permission denied".
export UV_PYTHON_INSTALL_DIR=/opt/vllm/uv-python
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
vllm download-kernels
log "vLLM installed (venv + kernels)"

# The venv and HF cache are created by root; make them readable so
# the ubuntu user can activate vllm and access the models.
chmod -R a+rX /opt/vllm

# 5. Serve shortcuts for the login user (ubuntu).
USER_HOME="/home/ubuntu"
mkdir -p "$USER_HOME"
# Shared prologue for all serve scripts: venv, shared HF cache, and the
# serving port (defined once here, used by every serve script).
cat > "$USER_HOME/serve.env" <<'EOF'
export HF_HOME=/opt/vllm/hf-cache
export SERVING_PORT=8000
source /opt/vllm/.venv/bin/activate
EOF
log "Writing $USER_HOME/serve-Qwen3.8-27B-FP8.sh"
cat > "$USER_HOME/serve-Qwen3.8-27B-FP8.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

source /home/ubuntu/serve.env

exec vllm serve Qwen/Qwen3.8-27B-FP8 \
--port "$SERVING_PORT" \
--max-model-len 262144 \
--max-num-seqs 128 \
--gpu-memory-utilization 0.92 \
--kv-cache-dtype fp8 \
--reasoning-parser qwen3 \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
EOF

log "Writing $USER_HOME/serve-Qwen3.8-27B-NVFP4.sh"
cat > "$USER_HOME/serve-Qwen3.8-27B-NVFP4.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

source /home/ubuntu/serve.env

# NVFP4 QAD checkpoint from Model Optimizer; requires --quantization modelopt.
exec vllm serve nvidia/Qwen3.8-27B-NVFP4-QAD \
--port "$SERVING_PORT" \
--quantization modelopt \
--max-model-len 262144 \
--max-num-seqs 128 \
--gpu-memory-utilization 0.92 \
--kv-cache-dtype fp8 \
--reasoning-parser qwen3 \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder
EOF

# Second script: small model for g4dn/T4 testing. The T4 has
# compute capability 7.5, so FlashAttention2 is unavailable
# (use the Triton backend) and the max context must fit the 16 GB
# VRAM left after the weights.
log "Writing $USER_HOME/serve-Qwen3-4B.sh"
cat > "$USER_HOME/serve-Qwen3-4B.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

source /home/ubuntu/serve.env

exec vllm serve Qwen/Qwen3-4B \
--port "$SERVING_PORT" \
--attention-backend TRITON_ATTN \
--max-model-len 24576 \
--gpu-memory-utilization 0.90 \
--enable-prefix-caching \
--enable-auto-tool-choice \
--tool-call-parser qwen3_coder \
--reasoning-parser qwen3
EOF
for f in serve.env serve-Qwen3.8-27B-FP8.sh serve-Qwen3.8-27B-NVFP4.sh serve-Qwen3-4B.sh; do
chmod +x "$USER_HOME/$f"
chown ubuntu:ubuntu "$USER_HOME/$f"
done

# Setup succeeded. Disable the reboot-continuation service so it does
# not start again on a future boot. Do NOT rm the unit file or
# daemon-reload here: this script is itself running under that service,
# and re-reading a now-missing unit mid-start makes systemd reset the
# running job to a default-timeout synthetic unit and kill it. Leaving
# the (now-disabled) file in place keeps the running job's definition
# intact; it will not start again because the enable symlink is gone.
if [ -f /etc/systemd/system/inference-setup.service ]; then
systemctl disable inference-setup.service || true
fi
log "Setup complete"
# 6. Optionally start serving the selected model now that setup is
# complete. $MODEL selects one of the serve scripts written above;
# "None" skips the launch.
if [ "${MODEL}" != "None" ]; then
# Run vLLM as its own supervised systemd service (not a nohup child):
# restarted on crash, logged to the journal, outlives this script.
log "Configuring vllm-serve.service for ${MODEL} (serve-${MODEL}.sh)"
# Unquoted heredoc on purpose: ${MODEL} is expanded by the shell at
# setup time (from params.env) and baked into the unit; the unit itself
# has no Environment= line, so leaving it quoted would make systemd
# resolve an empty variable at parse time (serve-.sh, exit 127).
cat > /etc/systemd/system/vllm-serve.service <<EOF
[Unit]
Description=vLLM inference server
After=network-online.target

[Service]
Type=simple
ExecStart=/bin/bash /home/ubuntu/serve-${MODEL}.sh
Restart=on-failure
RestartSec=30
TimeoutStartSec=infinity

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable vllm-serve.service
systemctl start vllm-serve.service
log "Started vllm-serve.service"
# Poll /health until the model is downloaded and loaded (can take many
# minutes on a first run); log progress every ~2.5 min so the wait is
# not mistaken for a hang. --max-time bounds each probe; curl -fs is
# non-zero while the server is down (set -e does not fire in the
# if-condition). SECONDS counts only this wait.
log "Waiting for vLLM to be ready (downloading model + loading into GPU); this can take many minutes"
VLLM_READY=0
VLLM_FAILED=0
SECONDS=0
for i in $(seq 1 240); do
if curl -fs --max-time 5 http://localhost:8000/health >/dev/null 2>&1; then
VLLM_READY=1
break
fi
if [ "$(systemctl show -p ActiveState --value vllm-serve.service 2>/dev/null)" = "failed" ]; then
VLLM_FAILED=1
log "ERROR: vllm-serve.service failed to start; see journalctl -u vllm-serve"
break
fi
if [ $((i % 15)) -eq 0 ]; then
log "vLLM not ready yet ($SECONDS s elapsed); still downloading/loading - see journalctl -u vllm-serve"
fi
sleep 10
done
if [ "$VLLM_READY" = "1" ]; then
log "vLLM ready to serve at http://localhost:8000 (took $SECONDS s)"
else
if [ "$VLLM_FAILED" = "0" ]; then
log "ERROR: vLLM did not become ready within timeout; see journalctl -u vllm-serve"
fi
fi
else
log "No model selected; vLLM installed but not started"
fi
