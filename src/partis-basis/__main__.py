from __future__ import annotations
import anyio
from anyio.streams.file import FileWriteStream, FileReadStream
import sys
import os
from pathlib import Path
import click
import fcntl
import termios
import tty
import pty
import select
import signal
import re
from tempfile import TemporaryDirectory
from shlex import quote
from subprocess import Popen
import shutil

venv = Path(os.environ['HOME'])/'.basis'

MPI_ENV_REC = re.compile('|'.join([
  # OpenMPI:
  '(OMPI_*\w+)',
  '(PMIX_*\w+)',
  '(OPAL_*\w+)',
  '(ORTE_*\w+)',

  # MPICH:
  '(MPIR_*\w+)',
  '(PMI_*\w+)',
  '(PMIX_*\w+)',
  '(HYDRA_*\w+)',

  # Intel MPI:
  '(I_MPI_*\w+)',
  '(MPIR_*\w+)',
  '(PMI_*\w+)',
  '(PMIX_*\w+)',
  '(HYDRA_*\w+)',

  # MVAPICH2:
  '(MV2_*\w+)',
  '(MPIR_*\w+)',
  '(PMI_*\w+)',
  '(PMIX_*\w+)',
  '(HYDRA_*\w+)']))

CONTAINER = "library://kcdodd/basis/ubuntu-22.04-jupyter:basis"

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def set_win_size(fd):
  if fd == -1:
    return

  # Get the window size of the real terminal
  winsize = fcntl.ioctl(sys.stdin.fileno(), termios.TIOCGWINSZ, b"\x00" * 8)
  # Set the window size of the pseudo-terminal
  fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def find_basis(start):
  tmp = start
  while tmp.name:
    if (tmp/'.basis').exists():
      return tmp/'.basis'

    tmp = tmp.parent

  return None

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
async def run(image, app, basis, work, cwd, cmd):

  if not shutil.which('singularity'):
    raise ValueError("Program 'singularity' not found")

  work = work.resolve()

  for start in [cwd, work]:
    if basis is not None:
      break

    basis = find_basis(start)

  if basis is None:
    raise ValueError("Basis directory must be given.")

  basis.mkdir(exist_ok=True, parents=True)
  basis = basis.resolve()

  print(f"image: {image}")
  print(f"  app: {app}")
  print(f"basis: {basis}")
  print(f" work: {work}")
  print(f"  cwd: {cwd}")
  print(f"  cmd: {' '.join(cmd)}")

  if cwd.is_relative_to(work):
    pwd = Path('/work')/cwd.relative_to(work)
  else:
    pwd = cwd

  assert app in ['bash', 'fish', 'jupyter', 'code']

  isatty = os.isatty(sys.stdin.fileno())

  if isatty:
    old_winch_handler = signal.getsignal(signal.SIGWINCH)
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())

  env = {
    k:v
    for k,v in os.environ.items()
    if MPI_ENV_REC.fullmatch(k)}

  with TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    env_file = tmp/'env_file.txt'

    env_file.write_text('\n'.join([
      f"{k}='{quote(str(v))}'"
      for k,v in env.items()]))

    try:
      pid, host_fd = pty.fork()
      if pid == 0:
        os.execvp('singularity', [
          'singularity',
          'run',
          '--app',
          app,
          '--cleanenv',
          '--no-home',
          '--env-file', str(env_file),
          '--pwd', str(pwd),
          '--bind', f"{basis}:/basis",
          '--bind', f"{work}:/work",
          str(image),
          *cmd])

        # should not return
        assert False

      if isatty:
        set_win_size(host_fd)
        signal.signal(signal.SIGWINCH, lambda signal, frame: set_win_size(host_fd))

      async def _fwd_io():
        stdin_fd = sys.stdin.fileno()
        stdout_fd = sys.stdout.fileno()

        while True:
          try:
            r, w, e = select.select([stdin_fd, host_fd], [], [])

            if stdin_fd in r:
              os.write(host_fd, os.read(stdin_fd, 1024))

            elif host_fd in r:
              os.write(stdout_fd, os.read(host_fd, 1024))

          except (OSError, EOFError):
            break

          await anyio.sleep(0)

      async with anyio.create_task_group() as tg:
        tg.start_soon(_fwd_io)

    finally:
      try:
        os.close(host_fd)
      except:
        pass

      host_fd = -1

      if isatty:
        signal.signal(signal.SIGWINCH, old_winch_handler)
        termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, old_tty)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@click.command(context_settings=dict(
  ignore_unknown_options=True,
))
@click.option(
  '-i', '--image',
  help = "",
  type = str,
  default = CONTAINER)
@click.option(
  '-a', '--app',
  help = "",
  type = str,
  default = 'fish')
@click.option(
  '-b', '--basis',
  help = "",
  type = Path,
  default = None)
@click.option(
  '-w', '--work',
  help = "",
  type = Path,
  default = Path.cwd())
@click.option(
  '-c', '--cwd',
  help = "",
  type = Path,
  default = Path.cwd())
@click.argument('cmd', nargs = -1, type = click.UNPROCESSED)
def main(image, app, basis, work, cwd, cmd):
  anyio.run(run, image, app, basis, work, cwd, cmd)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
exit(main())
