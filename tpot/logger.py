# -*- coding: utf-8 -*-

"""Copyright 2015-Present Randal S. Olson.

This file is part of the TPOT library.

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""
import datetime
from functools import wraps

from tqdm import tqdm


def pbar_check(func):
    """Decorate methods of Logger that depend on the _pbar attribute."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._pbar is not None and not self._pbar.disable:
            return func(self, *args, **kwargs)

    return wrapper


class Logger(object):
    """TPOT logger.

    Each log level defined in LOG_LEVELS is dynamically assigned a method.
    To utilize a specific log level, call the matching method, e.g.:

    logger.info('Lorem Ipsum.')

    Different logging techniques can be used with the `target` argument. The
    supported techniques are:
        * stdout - simply print the message
        * pbar - display the message through a tqdm progress bar.
    e.g.:

    logger.error('Something bad happened!', target='pbar')
    """

    LOG_LEVELS = ['none', 'error', 'warning', 'info']
    DISPLAY_TARGETS = ['stdout', 'pbar']

    @classmethod
    def _init_log_level_methods(cls):
        """Create methods for each log level above 'none'."""
        for level in cls.LOG_LEVELS[1:]:
            index = cls.LOG_LEVELS.index(level)

            def display_at_level(self, message, target='stdout', verbosity=index):
                self._display(message, target, verbosity)

            display_at_level.__name__ = level
            setattr(cls, level, display_at_level)

    def __init__(self, verbosity='none'):
        """Initialize a Logger.

        Parameters
        ----------
        verbosity : str
            The log level name.
        """
        self._pbar = None
        self._verbosity = self.LOG_LEVELS.index(verbosity)

    def init_pbar(self, *args, **kwargs):
        """Initialize the logger progress bar."""
        self._pbar = tqdm(*args, **kwargs)

    @pbar_check
    def update_pbar(self, amount):
        """Call update on the pbar object."""
        self._pbar.update(amount)

    def _display(self, message, target, verbosity):
        """Output a message.

        Parameters
        ----------
        message : str
            The message to display.
        target : str
            The display method.
        verbosity : int
            The log level required to display the message.
        """
        if self._verbosity >= verbosity:
            log_entry = '{LEVEL} :: {TIME} :: {MESSAGE}'.format(
                LEVEL=self.LOG_LEVELS[verbosity].upper(),
                TIME=datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
                MESSAGE=message
            )

            try:
                # Dynamically call the appropriate display method, which should
                # be named in the format of _display_{target}
                getattr(self, '_display_{}'.format(target))(log_entry)
            except AttributeError:
                raise ValueError('{} is not a valid logger target'.format(target))

    def _display_stdout(self, log_entry):
        """Display a log entry through printing it to stdout."""
        print(log_entry)

    @pbar_check
    def _display_pbar(self, log_entry):
        """Display a log entry through writing it to the progress bar."""
        self._pbar.write(log_entry)


Logger._init_log_level_methods()
