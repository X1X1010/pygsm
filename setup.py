from pathlib import Path

from setuptools import setup


def _read_version():
    namespace = {}
    version_path = Path(__file__).resolve().parent / 'pygsm' / '_version.py'
    exec(version_path.read_text(), namespace)
    return namespace['__version__']


if __name__ == '__main__':
    setup(version=_read_version())
