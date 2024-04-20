__import__("pkg_resources").declare_namespace(__name__)

__version__ = '0.0.1'

from .train import Trainer
from .flow  import Flow
