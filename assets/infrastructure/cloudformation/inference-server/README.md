# Inference Server

## Manual steps

Deploying the stack requires two resources that are created manually and passed as parameters:

- Create a managed prefix list containing the IPs that should be able to SSH into the instance, and use its `pl-…` id as the `AccessPrefixListId` parameter.
- Create a reusable Tailscale auth key and store it as an SSM SecureString parameter, and pass its name (without a leading `/`, e.g. `tailscale-auth-key`) as the `TailscaleAuthKeyParameterName` parameter.

## Modifying the setup script

The setup script lives in `scripts/inference-setup.sh` next to the template,
not in the template itself: in plaintext it exceeds EC2's 16 KiB user-data
limit, so the template's user-data is a small trampoline that decompresses a
gzip+base64 blob of the script and runs it (template parameters reach the
script via `/var/lib/inference-setup/params.env`).

After editing `scripts/inference-setup.sh`, regenerate the blob before
deploying the stack — the generator rewrites the blob in the template and
fails if the result exceeds the user-data limit:

```
python3 generate_userdata.py
```

The compressed blob leaves substantial headroom, so the script can keep
growing; if it ever exceeds it (tens of KB), move the script to a file
the instance fetches at boot instead.

## Model selection

The stack takes a `Model` parameter (a drop-down) that controls whether vLLM is started automatically at the end of setup:

- `None` (default): install everything, but don't start a server. SSH in and run a serve script manually.
- `Qwen3.8-27B-FP8`, `Qwen3.8-27B-NVFP4`, or `Qwen3-4B`: start the matching `serve-*.sh` right after setup finishes.

When a model is selected, vLLM runs as the supervised `vllm-serve` systemd service
(it is restarted on crash and survives reboots); its logs are covered in
[Logs](#logs). If it is not up, restart it with `sudo systemctl restart vllm-serve`.

## SSH

SSH in as the `ubuntu` user:

```
ssh ubuntu@<PUBLIC_IP> -i <PATH_TO_KEY>.pem
```

## Test the served model

```
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"<MODEL_ID>","messages":[{"role":"user","content":"ping"}]}'
```

Replace `<MODEL_ID>` with the model served by the script (or query `http://localhost:8000/v1/models`).

## Logs

### Web log viewer (on the tailnet)

The instance runs `inference-log-viewer`, a small HTTP service on port **8888**
that shows all of the logs below in a tabbed, auto-refreshing page — open
`http://ec2-inference-server.<tailnet>.ts.net:8888` in a browser to watch setup
progress from the tailnet without SSH. Each tab refreshes every 5 seconds;
the number of lines shown defaults to 500 and can be changed per tab
(2000 / 10000 / All) via the **Lines** selector, which is remembered
between visits. It has no authentication: assume anyone who can
reach the instance over the tailnet can read these logs (they contain setup
and service output only, no model input/output).

### High-level progress log

`/var/log/inference-setup.log` is the single file to check to know where setup is.
It records one timestamped line per key event, from the very first step of the
boot script (`Setup started`) to the terminal event (`vLLM ready to serve …`, or
`No model selected; …` when no model was requested). Because it is appended across
a possible NVIDIA-driver reboot, the first and last timestamps bracket the total
wall-clock time of the run — compare them to estimate how long the next run will
take.

```
sudo tail -n +1 -f /var/log/inference-setup.log
```

Each line looks like `2025-01-01T00:00:00Z [inference-setup] <event>`. A `vLLM ready
to serve` (or `No model selected; …`) line is the terminal success event; an
`ERROR:` line (e.g. vLLM did not become ready within timeout) is a terminal
failure. The verbose per-step output is still available below.

### Detailed logs

#### Cloud-init

The boot/setup script output is in:

```
sudo tail -n +1 -f /var/log/cloud-init-output.log
```

Cloud-init lifecycle and errors (including the "failed to run module" tracebacks):

```
sudo tail -n +1 -f /var/log/cloud-init.log
```

The script that ran:

```
sudo cat /var/lib/cloud/instance/scripts/part-001
```

#### Reboot fallback

If the NVIDIA driver required the fallback reboot, the setup script is re-run after the reboot by the `inference-setup` systemd service — its logs (instead of the cloud-init log) are in the journal:

```
sudo journalctl -u inference-setup --no-pager
```

or follow the setup live as it runs (Ctrl-C to stop):

```
sudo journalctl -u inference-setup -f
```

#### vLLM server

vLLM server logs (when a model is selected) are in the `vllm-serve` service:

```
sudo journalctl -u vllm-serve -f
```
