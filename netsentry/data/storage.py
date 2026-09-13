import os
import sys
import boto3
from botocore.config import Config
from dotenv import load_dotenv
load_dotenv()


class ProgressPercentage:
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        percentage = (self._seen_so_far / self._size) * 100
        sys.stdout.write(
            f"\rUploading {self._filename}: {self._seen_so_far / (1024*1024):.2f}MB / {self._size / (1024*1024):.2f}MB ({percentage:.1f}%)"
        )
        sys.stdout.flush()

def upload_file_to_r2(local_path: str, r2_path: str):
    r2 = boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )
    bucket = os.getenv("R2_BUCKET_NAME", "netsentry")
    print(f"Connecting to Cloudflare R2 bucket '{bucket}'...")
    r2.upload_file(local_path, bucket, r2_path, Callback=ProgressPercentage(local_path))
    print(f"\nSuccessfully uploaded to {bucket}/{r2_path}!")


def download_file_from_r2(r2_path: str, local_path: str):
    """Downloads a file directly from Cloudflare R2 using R2_* environment variables."""
    r2 = boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )
    bucket = os.getenv("R2_BUCKET_NAME", "netsentry")
    print(f"Downloading from Cloudflare R2 bucket '{bucket}'...")
    r2.download_file(bucket, r2_path, local_path, Callback=ProgressPercentage(local_path))
    print(f"Successfully downloaded {bucket}/{r2_path} -> {local_path}")