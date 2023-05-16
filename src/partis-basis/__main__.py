from __future__ import annotations
import anyio
from anyio import Path as APath
import sys
import os
from pathlib import Path
import click
import fcntl
import termios
import tty
import pty
import signal
import re
from tempfile import TemporaryDirectory
from shlex import quote
import shutil
import webbrowser
import asks
import errno
import shlex
from timeit import default_timer as timer
import json

venv = Path(os.environ['HOME'])/'.basis'
APPS = ['bash', 'fish', 'jupyter', 'code']

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

CONTAINER = "library://kcdodd/basis/ubuntu-22.04-basis:0.0.2"

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
async def check_code_server_and_open_browser(url, token_file):
  await anyio.sleep(2)

  while True:
    if await token_file.exists():
      token = (await token_file.read_text()).strip()
      _url = f"{url}/?tkn={token}"
      print(_url)

      try:
        response = await asks.get(_url)
        webbrowser.open(_url)
        break

      except Exception as e:
        # print(f"GET {_url} -> {e}")
        # sys.stdout.flush()
        # import traceback
        # traceback.print_exc()
        pass

    await anyio.sleep(2)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
async def check_jupyter_server_and_open_browser(url, token_dir):
  await anyio.sleep(2)

  while True:
    if await token_dir.exists():
      files = []

      async for file in token_dir.glob('jpserver-*.json'):
        files.append(file)

      if not files:
        continue

      config = json.loads(await files[-1].read_text())

      token = config.get('token', '')
      port = config.get('port', '')
      _url = f"{url}:{port}/lab?token={token}"

      try:
        response = await asks.get(_url)
        webbrowser.open(_url)
        break

      except Exception as e:
        # print(f"GET {_url} -> {e}")
        # sys.stdout.flush()
        # import traceback
        # traceback.print_exc()
        pass

    await anyio.sleep(2)

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

  assert app in APPS

  jupyter_dir = basis/'.local'/'share'/'jupyter'/'runtime'

  if jupyter_dir.exists():
    shutil.rmtree(str(jupyter_dir))

  print("Entering container...")

  isatty = os.isatty(sys.stdin.fileno())
  restore_io = True
  host_fd = -1
  pid = 0

  stdin_fd = sys.stdin.fileno()
  stdout_fd = sys.stdout.fileno()

  stdin_fl = fcntl.fcntl(stdin_fd, fcntl.F_GETFL)

  fcntl.fcntl(stdin_fd, fcntl.F_SETFL, stdin_fl | os.O_NONBLOCK)

  if isatty:
    old_winch_handler = signal.getsignal(signal.SIGWINCH)
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setraw(stdin_fd)

  def _restore_io():
    nonlocal restore_io, host_fd, pid

    if restore_io:
      restore_io = False

      if pid != 0:
        os.kill(pid, signal.SIGKILL)
        pid = 0

      fcntl.fcntl(stdin_fd, fcntl.F_SETFL, stdin_fl)

      try:
        os.close(host_fd)
      except:
        pass

      host_fd = -1

      if isatty:
        signal.signal(signal.SIGWINCH, old_winch_handler)
        termios.tcsetattr(stdin_fd, termios.TCSAFLUSH, old_tty)

      print("... Exited container")

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

    cmd_singularity = [
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
      *cmd]

    try:
      pid, host_fd = pty.fork()
      if pid == 0:
        os.execvp(cmd_singularity[0], cmd_singularity)

        # should not return
        assert False

      fcntl.fcntl(host_fd, fcntl.F_SETFL, fcntl.fcntl(host_fd, fcntl.F_GETFL)|os.O_NONBLOCK)

      if isatty:
        set_win_size(host_fd)
        signal.signal(signal.SIGWINCH, lambda signal, frame: set_win_size(host_fd))

      async def _fwd_io():
        try:
          ti = timer()
          while True:
            tf = timer()
            await anyio.sleep(max(0, 0.025 - (tf-ti)))
            ti = timer()

            try:
              os.write(host_fd, os.read(stdin_fd, 1024))
            except OSError as e:
              if e.errno not in [errno.EAGAIN, errno.EWOULDBLOCK]:
                raise
            except EOFError:
              raise

            try:
              os.write(stdout_fd, os.read(host_fd, 1024))
            except OSError as e:
              if e.errno not in [errno.EAGAIN, errno.EWOULDBLOCK]:
                raise
            except EOFError:
              raise

        finally:
          _restore_io()

      async with anyio.create_task_group() as tg:
        tg.start_soon(_fwd_io)

        if app == 'code':
          token_file = APath(basis/'.openvscode-server'/'data'/'token')
          tg.start_soon(
            check_code_server_and_open_browser,
            "http://localhost:3000",
            token_file)

        elif app == 'jupyter':
          tg.start_soon(
            check_jupyter_server_and_open_browser,
            "http://localhost",
            APath(jupyter_dir))

    except OSError as e:
      if e.errno == errno.EIO:
        pass
    finally:
      _restore_io()

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
  help = f"{{{', '.join(APPS)}}}",
  type = click.Choice(APPS),
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
  default = Path('/'))
@click.option(
  '-c', '--cwd',
  help = "",
  type = Path,
  default = Path.cwd())
@click.argument(
  'cmd',
  nargs = -1,
  type = click.UNPROCESSED)
def main(image, app, basis, work, cwd, cmd):
  anyio.run(
    run, image, app, basis, work, cwd, cmd,
    backend_options = dict(
      use_uvloop = True))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
exit(main())
