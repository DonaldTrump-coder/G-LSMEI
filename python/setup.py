from skbuild import setup

setup(
    name="lsmatching",
    version="0.0.1",
    cmake_source_dir=".",
    cmake_args=["-G", "Visual Studio 17 2022", "-A", "x64"],
    packages=["lsmatching"],
    package_dir={"lsmatching": "lsmatching"},
    package_data={"lsmatching": ["_lsmatching*.pyd", "__init__.py", "*.dll"]},
    include_package_data=True,
)