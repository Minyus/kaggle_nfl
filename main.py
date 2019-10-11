import sys
import os
from pathlib import Path

project_path = Path(__file__).resolve().parent

src_path = project_path / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

if "PYTHONPATH" not in os.environ:
    os.environ["PYTHONPATH"] = src_path

from kedex.mlflow_context import MLflowFlexibleContext


context = MLflowFlexibleContext(project_path)
context.run()
