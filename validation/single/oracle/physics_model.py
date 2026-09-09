"""Compatibility import for the original laser oracle file path."""
import importlib.util
import sys
from pathlib import Path

_name = "_validation_laser_compat"
_folder = Path(__file__).resolve().parents[1] / "laser_welding"
if _name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _name, _folder / "physics_model.py", submodule_search_locations=[str(_folder)]
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _module
    _spec.loader.exec_module(_module)
else:
    _module = sys.modules[_name]

ModelParameters = _module.ModelParameters
evaluate_model = _module.evaluate_model
inclusive_grid = _module.inclusive_grid
sigmoid = _module.sigmoid
generate_grid = _module.generate_grid
plot_grid = _module.plot_grid
build_parser = _module.build_parser
main = _module.main

if __name__ == "__main__":
    main()
