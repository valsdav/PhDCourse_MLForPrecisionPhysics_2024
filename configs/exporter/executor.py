from pocket_coffea.executors.executors_cern_swan import DaskExecutorFactory
from dask.distributed import WorkerPlugin, Worker, Client
import importlib.util
from dask.distributed import WorkerPlugin

class PackageChecker(WorkerPlugin):
    def __init__(self, package_name):
        self.package_name = package_name

    def setup(self, worker):
        try:
            spec = importlib.util.find_spec(self.package_name)
            if spec is None:
                raise ImportError(f"Package {self.package_name} not found")
        except ImportError:
           # worker.log.error(f"Package {self.package_name} not found. Restarting worker...")
            worker.close()
            raise SystemExit(1)


class ExecutorFactory(DaskExecutorFactory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        super().setup()
        # now setting up and registering the ONNX plugin
       
        # Registering the session plugin
        self.dask_client.register_worker_plugin(PackageChecker("pocket_coffea"))


def get_executor_factory(executor_name, **kwargs):
    return ExecutorFactory(**kwargs)
