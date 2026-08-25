"""Environment-backed Celery application imported by the worker CLI."""

from mindbridge.consolidation_worker import register_consolidation_schedule
from mindbridge.worker import WorkerSettings, create_worker_app

app = create_worker_app(WorkerSettings.from_environment())

# A no-op unless MINDBRIDGE_CONSOLIDATION_TENANT_IDS names tenants, and even then it only adds
# a task on its own queue plus a beat entry, so an observation worker started against this app
# behaves exactly as it did before.
register_consolidation_schedule(app)
