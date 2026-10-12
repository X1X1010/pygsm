import logging

import pytest

from pygsm.utils import output


def _remove_installed_handler():
    if output._handler is not None:
        output.logger.removeHandler(output._handler)
        output._handler = None


@pytest.fixture
def restore_logger():
    """Start without the handler a previous ``setup_logging`` call (e.g. a CLI test) installed."""
    level = output.logger.level
    _remove_installed_handler()
    yield
    _remove_installed_handler()
    output.logger.setLevel(level)


def test_library_is_silent_without_setup(capsys, restore_logger):
    logging.getLogger('pygsm.gsm.base').warning('not shown')
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''


def test_setup_logging_prints_bare_messages_at_level(capsys, restore_logger):
    output.setup_logging()
    log = logging.getLogger('pygsm.optimizers.lbfgs')
    log.info(' Node: 1 Opt step: 1')
    log.debug('hidden')
    assert capsys.readouterr().out == ' Node: 1 Opt step: 1\n'

    output.setup_logging(logging.DEBUG)  # replaces the previous handler
    log.debug('shown')
    assert capsys.readouterr().out == 'shown\n'


def test_setup_logging_to_file(tmp_path, restore_logger):
    path = tmp_path / 'gsm.log'
    output.setup_logging(filename=str(path))
    output.printcool('Starting opt iter 3')
    output._handler.flush()
    lines = path.read_text().splitlines()
    assert lines[0] == lines[2] == '#' + '=' * 56 + '#'
    assert lines[1] == f'#|  {"Starting opt iter 3":^50}  |#'
