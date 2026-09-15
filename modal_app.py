import modal

image = (
    modal.Image.from_dockerfile("docker/Dockerfile.server")
    .add_local_dir("netsentry", remote_path="/app/netsentry")
    .add_local_file("configs/serving.yaml", remote_path="/app/configs/serving.yaml")
)

app = modal.App(name="netsentry-serving", image=image)

# Load secrets directly from local .env
env_secret = modal.Secret.from_dotenv()

@app.function(
    secrets=[env_secret],
    min_containers=1,  # Keep 1 instance permanently warm to eliminate cold starts
    scaledown_window=600,  # Keep scaled instances warm for 10 minutes
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    """Mounts NetSentry v2 FastAPI serving application directly into Modal."""
    from netsentry.serving.app import app as web_app
    return web_app
