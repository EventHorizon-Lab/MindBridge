"""Environment-backed Celery application imported by the worker CLI."""

from mindbridge.worker import WorkerSettings, create_worker_app

app = create_worker_app(WorkerSettings.from_environment())
