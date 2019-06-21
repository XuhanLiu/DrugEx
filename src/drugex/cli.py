# -*- coding: utf-8 -*-

"""The command line interface for ``drugex``.

Why does this file exist, and why not put this in ``__main__``?

You might be tempted to import things from ``__main__`` later, but that will cause
problems--the code will get executed twice:

- When you run `python3 -m drugex` python will execute
  ``__main__.py`` as a script. That means there won't be any
  ``drugex.__main__`` in ``sys.modules``.
- When you import ``__main__`` it will get executed again (as a module) because
  there's no ``drugex.__main__`` in ``sys.modules``.

Also see https://click.pocoo.org/latest/setuptools/
"""

import click

import drugex.dataset
import drugex.environ
import drugex.pretrainer
import drugex.agent
import drugex.designer

__all__ = ['main']

main = click.Group(commands={
    'dataset': drugex.dataset.main,
    'environ': drugex.environ.main,
    'pretrainer': drugex.pretrainer.main,
    'agent': drugex.agent.main,
    'designer': drugex.designer.main
})

if __name__ == '__main__':
    main()
