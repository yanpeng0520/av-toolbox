# Public Demo on DGX Spark with Cloudflare Tunnel

This workflow keeps GitHub as the project entry point while the DGX Spark runs
all Python, FFmpeg, `av_toolbox`, and optional GPU/DenseAV work:

```text
GitHub README -> https://demo.yan-peng.com -> Cloudflare Tunnel -> Streamlit on DGX Spark -> av_toolbox + GPU
```

Cloudflare Tunnel is the preferred public edge because `cloudflared` connects
outbound from the DGX to Cloudflare. The Streamlit app can stay bound to
`127.0.0.1`, with no inbound port opened on the DGX.

## 1. Install the App on the DGX

```bash
cd /home/yanp/projects/av_toolbox
python -m pip install --upgrade pip
python -m pip install -e ".[audio,video,av,web]"
```

For DenseAV/GPU runs, install the heavier dependency set and place weights in
the av-toolbox cache:

```bash
python -m pip install -e ".[denseav]"
mkdir -p ~/.cache/av_toolbox/weights
# Expected default names:
# ~/.cache/av_toolbox/weights/denseav_2head.ckpt
# ~/.cache/av_toolbox/weights/denseav_sound.ckpt
```

## 2. Start Public Demo Mode Locally

CPU/lightweight public demo:

```bash
av-toolbox serve \
  --host 127.0.0.1 \
  --port 8501 \
  --output-root /srv/av-toolbox-demo/runs \
  --page-title "AV Toolbox Demo" \
  --public-demo \
  --public-max-seconds 20 \
  --public-max-upload-mb 100
```

DGX/GPU demo with DenseAV available in the workflow picker:

```bash
av-toolbox serve \
  --host 127.0.0.1 \
  --port 8501 \
  --output-root /srv/av-toolbox-demo/runs \
  --page-title "AV Toolbox Demo" \
  --public-demo \
  --public-enable-denseav \
  --public-max-seconds 10 \
  --public-max-upload-mb 100
```

Public mode starts on a bundled sample clip and also accepts uploads. It
intentionally removes arbitrary media paths, output paths, cache paths,
checkpoints, and advanced runtime controls from the browser. Public visitors can
only change bounded demo settings such as analyzed duration and overlay export.
Uploaded files are capped, each run gets a server-side output directory, and
public runs are serialized in the Streamlit process.

## 3. Run Streamlit as a Service

Copy and adapt the template:

```bash
sudo cp deploy/systemd/av-toolbox-public-demo.service.example /etc/systemd/system/av-toolbox-public-demo.service
sudo systemctl daemon-reload
sudo systemctl enable --now av-toolbox-public-demo
sudo systemctl status av-toolbox-public-demo
```

Edit these values before starting the service:

- `WorkingDirectory`: checkout path on the DGX
- `ExecStart`: Python environment path if `av-toolbox` is not on the system PATH
- `--output-root`: durable run/output directory
- `AV_TOOLBOX_PUBLIC_ENABLE_DENSEAV`: set to `1` only after DenseAV is installed and weights exist

## 4. Create the Cloudflare Tunnel

Dashboard path:

1. Cloudflare dashboard -> Zero Trust -> Networks -> Tunnels.
2. Create a tunnel for the DGX Spark.
3. Install/run the generated `cloudflared` command on the DGX.
4. Add a public hostname:
   - Hostname: `demo.yan-peng.com`
   - Service: `http://localhost:8501`

CLI/config-file path:

```bash
cloudflared tunnel login
cloudflared tunnel create av-toolbox-demo
cloudflared tunnel route dns av-toolbox-demo demo.yan-peng.com
```

Then adapt [`deploy/cloudflare/cloudflared-config.example.yml`](../deploy/cloudflare/cloudflared-config.example.yml)
and install `cloudflared` as a service for that config.

## 5. Add Access and Limits

For a private beta, put Cloudflare Access in front of `demo.yan-peng.com` and
allow only selected emails or identity providers. For a public launch, keep the
app in public-demo mode and use Cloudflare WAF/rate limiting to reduce abuse.

Upload sizing should stay at or below your Cloudflare plan limit. A conservative
public default is 100 MB and 20 seconds; DenseAV demos should usually be shorter.

## 6. Link from GitHub

Add the public link near the top of the README:

```markdown
Live demo: https://demo.yan-peng.com
```

Use the README link as the entry point; Cloudflare handles HTTPS and routing,
and the DGX remains the origin compute host.

## References

- Cloudflare Tunnel: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/
- Cloudflare Access self-hosted apps: https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/self-hosted-public-app/
- Cloudflare upload size limits: https://developers.cloudflare.com/support/troubleshooting/http-status-codes/4xx-client-error/error-413/
