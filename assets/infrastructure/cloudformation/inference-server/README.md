# Inference Server

## Manual steps

Deploying the stack requires two resources that are created manually and passed as parameters:

- Create a managed prefix list containing the IPs that should be able to SSH into the instance, and use its `pl-…` id as the `AccessPrefixListId` parameter.
- Create a reusable Tailscale auth key and store it as an SSM SecureString parameter, and pass its name (without a leading `/`, e.g. `tailscale-auth-key`) as the `TailscaleAuthKeyParameterName` parameter.

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

From inside the instance, the boot/setup script output is in:

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

If the NVIDIA driver required the fallback reboot, the setup script is re-run after the reboot by the `inference-setup` systemd service — its logs (instead of the cloud-init log) are in the journal:

```
sudo journalctl -u inference-setup --no-pager
```

or follow the setup live as it runs (Ctrl-C to stop):

```
sudo journalctl -u inference-setup -f
```
