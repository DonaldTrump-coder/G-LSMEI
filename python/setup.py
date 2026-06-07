from skbuild import setup
from pathlib import Path

version_path = Path("package/version.py")
version = {}
exec(version_path.read_text(), version)

setup(
    name="glsmei",
    version=version["__version__"],
    cmake_source_dir=".",
    cmake_args=["-G", "Visual Studio 17 2022", "-A", "x64"],
    packages=["lsmatching"],
    package_dir={"lsmatching": "package"},
    package_data={"lsmatching": ["_lsmatching*.pyd", "__init__.py", "version.py", "*.dll"]},
    include_package_data=True,
    long_description=Path("package/README_PyPI.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
)