from __future__ import annotations
import sys
import os
from pathlib import Path
import re
import shutil
import json
import ctypes
import tempfile
import argparse
import shlex
from subprocess import check_output, check_call, CalledProcessError, STDOUT

#===============================================================================
_PY = f"{'.'.join(str(v) for v in sys.version_info[:2])}"
_CONFIG_DIR = Path.home()/'.config'/'partis-basis'
_CONFIG_FILE = _CONFIG_DIR/'config.json'

#===============================================================================
# directory for Container Device Interface (CDI) configs
_CDI_DIR = _CONFIG_DIR/'cdi'
# NOTE: '/etc/cdi' and '/var/run/cdi' are managed by the host (root), only specs
# in _CDI_DIR are generated/updated here
_CDI_DIRS = [_CDI_DIR, Path('/etc/cdi'), Path('/var/run/cdi')]
_CDI_NVIDIA = 'nvidia.yaml'

#===============================================================================
_CONFIG = {}
_CONFIG_DEFAULTS = {
  "default_image": "library://kcdodd/basis/ubuntu-22.04-basis:0.2.0",
  # 'auto' -> generate nvidia CDI spec on demand, 'off' -> never generate
  "nvidia_cdi": "auto"}

if _CONFIG_FILE.exists():
  _CONFIG = json.loads(_CONFIG_FILE.read_bytes())

_CONFIG = _CONFIG_DEFAULTS|_CONFIG

#===============================================================================
APPS = ['bash', 'fish', 'claude', 'manage']

PASS_ENV = frozenset([
  'PATH',
  'LD_LIBRARY_PATH',
  'USER',
  'LOGNAME',
  'LANG',
  'LANGUAGE',
  'LC_CTYPE',
  'TERM',
  'COLORFGBG',
  'COLORTERM',
  'LSCOLORS',
  'LS_COLORS',
  'DISPLAY',
  'NO_COLOR',
  'PY_COLORS',
  'CLICOLOR',
  'CLICOLOR_FORCE',
  'FORCE_COLOR',
  'NOHM_MIN_LOG_LEVEL',
  'BASIS_DEBUG',
  'XLA_PYTHON_CLIENT_ALLOCATOR',
  'XLA_PYTHON_CLIENT_PREALLOCATE',
  'TF_CPP_MIN_LOG_LEVEL',
  'TF_CPP_MAX_VLOG_LEVEL',
  'TF_CPP_VMODULE',
  'PJRT_NPROC',
  'XLA_FLAGS',
  # pass claude pre-auth for headless runs
  'CLAUDE_CODE_OAUTH_TOKEN'])

MPI_ENV_REC = re.compile('|'.join([
  # OpenMPI:
  r'(OMPI_*\w+)',
  r'(PMIX_*\w+)',
  r'(OPAL_*\w+)',
  r'(ORTE_*\w+)',

  # MPICH:
  r'(MPIR_*\w+)',
  r'(PMI_*\w+)',
  r'(PMIX_*\w+)',
  r'(HYDRA_*\w+)',

  # Intel MPI:
  r'(I_MPI_*\w+)',
  r'(MPIR_*\w+)',
  r'(PMI_*\w+)',
  r'(PMIX_*\w+)',
  r'(HYDRA_*\w+)',

  # MVAPICH2:
  r'(MV2_*\w+)',
  r'(MPIR_*\w+)',
  r'(PMI_*\w+)',
  r'(PMIX_*\w+)',
  r'(HYDRA_*\w+)']))

#===============================================================================
def _debug() -> bool:
  return bool(os.environ.get('BASIS_DEBUG'))

#===============================================================================
def _tail(output: bytes|None, n: int = 5) -> list[str]:
  lines = (output or b'').decode('utf-8', errors='replace').splitlines()

  return [line for line in lines if line.strip()][-n:]

#===============================================================================
def _nvidia_driver() -> str|None:
  """Version of the host nvidia kernel driver, or None if no driver is present

  Tests for the driver itself, not the container toolkit. 'libnvidia-ml.so.1'
  is exactly what 'nvidia-ctk' needs to enumerate devices.
  """
  try:
    ctypes.CDLL('libnvidia-ml.so.1')
  except OSError:
    return None

  # native device nodes, or WSL2 (driver libraries under '/usr/lib/wsl/lib')
  if not (Path('/dev/nvidiactl').exists() or Path('/dev/dxg').exists()):
    return None

  version = Path('/sys/module/nvidia/version')

  # NOTE: not present on WSL2, where the spec is not pinned to a host driver
  return version.read_text().strip() if version.exists() else 'unknown'

#===============================================================================
def _nvidia_cdi(device: str, *, refresh: bool = False) -> int:
  """Ensure a Container Device Interface (CDI) spec exists for an nvidia device

  Returns non-zero if the requested device cannot be made available.
  """
  if _CONFIG.get('nvidia_cdi') == 'off':
    # host is assumed to manage CDI specs
    return 0

  driver = _nvidia_driver()

  if driver is None:
    print(
      f"Error: --device {device!r} requires an nvidia driver"
      " ('libnvidia-ml.so.1' not found)")
    return 1

  nvidia_ctk = shutil.which('nvidia-ctk')

  if nvidia_ctk is None:
    print(
      f"Error: --device {device!r} requires 'nvidia-ctk'"
      " (nvidia-container-toolkit)")
    return 1

  cdi = _CDI_DIR/_CDI_NVIDIA
  # generated spec references version suffixed driver libraries, so it must be
  # re-generated after a host driver upgrade
  stamp = _CDI_DIR/f'{_CDI_NVIDIA}.driver'
  stale = not (
    cdi.exists()
    and stamp.exists()
    and stamp.read_text().strip() == driver)

  for _dir in _CDI_DIRS[1:]:
    if (_dir/_CDI_NVIDIA).exists():
      # prefer the host managed spec, but do not let a stale one shadow it
      if stale and cdi.exists():
        cdi.unlink()

      return 0

  if not (stale or refresh):
    return 0

  _CDI_DIR.mkdir(exist_ok=True, parents=True)

  # Generate out of place so a failed run cannot leave a truncated spec, but not
  # within _CDI_DIR itself, since anything there is a spec as far as the runtime
  # is concerned.
  # NOTE: nvidia-ctk infers the spec format from the output file extension,
  # appending '.yaml' if it is not recognized.
  fd, _tmp = tempfile.mkstemp(
    dir=_CDI_DIR.parent,
    prefix=f'.{_CDI_NVIDIA}.',
    suffix='.yaml')

  os.close(fd)
  tmp = Path(_tmp)

  cmd = [nvidia_ctk, 'cdi', 'generate', f'--output={tmp}']

  try:
    # nvidia-ctk is chatty on stderr even when it succeeds
    output = check_output(cmd, stderr=STDOUT)

  except (CalledProcessError, OSError) as e:
    tmp.unlink(missing_ok=True)

    print(
      "Error: Could not create nvidia Container Device Interface (CDI)"
      f" config: {e}")

    for line in _tail(getattr(e, 'output', None)):
      print(f"  {line}")

    print(f"  command: {shlex.join(cmd)}")
    return 1

  if not (tmp.is_file() and tmp.stat().st_size):
    tmp.unlink(missing_ok=True)

    print(
      "Error: nvidia Container Device Interface (CDI) config not generated:"
      f" {tmp}")

    for line in _tail(output):
      print(f"  {line}")

    print(f"  command: {shlex.join(cmd)}")
    return 1

  os.replace(tmp, cdi)
  stamp.write_text(driver)

  if _debug():
    print(f"Generated CDI config for nvidia driver {driver}: {cdi}")

    for line in _tail(output, 100):
      print(f"  {line}")

  return 0

#===============================================================================
def _basis(*,
    image: str|None = None,
    basis: Path|None = None,
    create: bool = True) -> tuple[str, Path, Path]:

  if basis is None:
    venv: str|Path = os.environ.get('VIRTUAL_ENV', '')

    if venv:
      venv = Path(venv).resolve()

    if not (venv and venv.exists()):
      raise FileNotFoundError(f"No active virtual environment: {venv}")

    basis = venv/'.basis'

  default = basis/'mods'/'environment'/'.version'
  created = False

  if create and not basis.exists():
    basis.mkdir(parents=True)
    created = True

  basis_config_file = basis/'.config'/'basis'

  if basis_config_file.exists():
    basis_config = json.loads(basis_config_file.read_text())
    _image = basis_config['image']

    if image is None:
      # image originally used to create basis directory
      image = _image

    elif image != _image:
      print("Warning: Basis directory originally created using different image, behavior may be undefined.")
      print(f" original imag: {_image!r}")

  else:
    basis_config_file.parent.mkdir(exist_ok=True, parents=True)

    if created:
      if image is None:
        image = _CONFIG.get('default_image')

      basis_config_file.write_text(json.dumps({
        'image': str(image)}))

    else:
      # fixup environment created without config file
      # NOTE: hardcoded assumption that all older versions created with this image
      basis_config_file.write_text(json.dumps({
        'image': "library://kcdodd/basis/ubuntu-22.04-basis:0.0.2"}))

  assert isinstance(image, str)

  return image, basis, default

#===============================================================================
def init(*, image, basis, cwd, python):
  name = 'init'
  pwd = cwd.resolve()

  singularity = shutil.which('singularity')

  if not singularity:
    print("Error: Program 'singularity' not found")
    return 1

  image, basis, default = _basis(basis=basis, image=image)
  env_dir = basis/'envs'/name

  if env_dir.is_dir() and any(env_dir.iterdir()):
    print(f"Cannot initialize non-empty directory: {env_dir}")
    return 1

  env = {
    k:v
    for k,v in os.environ.items()
    if k in PASS_ENV or MPI_ENV_REC.fullmatch(k)}

  def _run(*cmd):
    check_call([
      singularity,
      'run',
      '--app', 'manage',
      '--no-home',
      '--pwd', str(pwd),
      '--bind', f"{basis}:/basis",
      str(image),
      *cmd],
      env=env)

  _run('create', name, str(python))

  default.write_text(f'#%Module\nset ModulesVersion {name}')

  return 0

#===============================================================================
def run(*,
    image,
    basis,
    app,
    read,
    work,
    host_read,
    cwd,
    cmd,
    device,
    oci_mode,
    refresh_cdi,
    silent,
    network,
    knowledge):
  singularity = shutil.which('singularity')

  # nvccli = shutil.which('nvidia-container-cli')

  if not singularity:
    print("Error: Program 'singularity' not found")
    return 1

  # _owner = Path(singularity).owner()

  # if _owner != 'root' and _owner == os.getlogin():
  #   oci_mode = True

  # if nvccli:
  #   runhelp = check_output([singularity, 'run', '--help']).decode('utf-8')

  #   if '--nvccli' not in runhelp:
  #     print("Warning: singularity version must be >= 3.9 to utilize nvidia-container-toolkit found on host")
  #     nvccli = None

  pwd = cwd.resolve()
  work += [pwd]

  image, basis, default = _basis(basis=basis, image=image)

  env = {
    k:v
    for k,v in os.environ.items()
    if k in PASS_ENV or MPI_ENV_REC.fullmatch(k)}

  extra_args = []

  # if nvccli:
  #   # generate
  #   extra_args.extend(['--nv', '--nvccli', '--oci'])

  if device:
    # currently CDI device only supported in OCI mode
    oci_mode = True

    if device.startswith('nvidia.com/') and _nvidia_cdi(device, refresh=refresh_cdi):
      return 1

    extra_args.extend([
      '--device', device])

  if oci_mode:
    # singularity requires each search directory to exist
    _CDI_DIR.mkdir(exist_ok=True, parents=True)

    extra_args.extend([
      '--oci',
      '--no-compat',
      '--cdi-dirs', ','.join(str(v) for v in _CDI_DIRS)])


  if network:
    extra_args.extend([
      '--net',
      '--network', network])

  mounts = [
    # '--bind', f"{basis}:/basis"
    '--mount', f'type=bind,"src={basis}",dst=/basis']

  for path in read:
    path = str(path.resolve())
    mounts.extend([
      # '--bind', f"{path}:ro"
      '--mount', f'type=bind,"src={path}","dst={path}",ro'])

  for path in work:
    path = str(path.resolve())
    mounts.extend([
      #'--bind', str(path)
      '--mount', f'type=bind,"src={path}","dst={path}"'])

  if knowledge:
    env['OKF_ROOT'] = '/okf'

    mounts.extend([
      # '--bind', f"{knowledge}:/okf"
      '--mount', f'type=bind,"src={knowledge.resolve()}",dst=/okf'])

  if host_read:
    mounts.extend([
      '--mount', 'type=bind,src=/,dst=/host,ro'])

  cmd_singularity = [
    singularity,
    'run',
    '--app',
    app,
    *extra_args,
    '--no-home',
    '--pwd', str(pwd),
    *mounts,
    str(image),
    *cmd]

  if not silent:
    print(f"image: {image}")
    print(f"  app: {app}")
    print(f"basis: {basis}")

    if host_read:
      print(" host: (read-only) '/' -> '/host'")

    if read:
      print(f" read: {', '.join([repr(str(path)) for path in read])}")

    print(f" work: {', '.join([repr(str(path)) for path in work])}")
    print(f"  cwd: {repr(str(cwd))}")

    if cmd:
      print(f" exec: {shlex.join(cmd)}")

    print(f"Running: {shlex.join(cmd_singularity[1:])}")

    # flush stdout before replacing process
    sys.stdout.flush()
    sys.stderr.flush()

  os.execve(cmd_singularity[0], cmd_singularity, env)

#===============================================================================
def _cuda_info(nvccli):
  info = check_output([nvccli, 'info', '--csv']).decode('utf-8')
  lines = info.splitlines()
  nvrm_version, cuda_version = lines[1].split(',')

  nvrm_version = tuple(int(v) for v in nvrm_version.split('.'))
  cuda_version = tuple(int(v) for v in cuda_version.split('.'))

  return nvrm_version, cuda_version

#===============================================================================
def main():

  parser = argparse.ArgumentParser(
    description="partis-basis runner for singularity container based environments")

  parser.add_argument(
    '-i', '--image',
    help="",
    type=str,
    default=None)

  parser.add_argument(
    '-a', '--app',
    help=f"{', '.join(APPS)}",
    # choices=APPS, # too restrictive, can run other containers with different apps (fail at run)
    default='bash')

  parser.add_argument(
    '-b', '--basis',
    help="",
    type=Path,
    default=None)

  parser.add_argument(
    '--init',
    dest='init',
    action='store_true',
    help="Create and initialize basis directory if it doesn't exist")

  parser.add_argument(
    '--oci',
    dest='oci_mode',
    action='store_true',
    help="Run container in OCI mode")

  parser.add_argument(
    '--refresh-cdi',
    dest='refresh_cdi',
    action='store_true',
    help="Re-generate the Container Device Interface (CDI) config for --device")

  parser.add_argument(
    '--host_read',
    dest='host_read',
    action='store_true',
    help="Add a read-only mount of host system at /host")

  parser.add_argument(
    '--silent',
    dest='silent',
    action='store_true',
    help="Run without reporting container startup")

  parser.add_argument(
    '--network',
    type=str,
    default = '',
    help="Run in a new network")

  parser.add_argument(
    '--python',
    type=str,
    default=_PY)

  parser.add_argument(
    '--device',
    type=str,
    default=None)

  parser.add_argument(
    '-r', '--read',
    help="Add a read-only directory to mounts",
    type=Path,
    action='append',
    default=[])

  parser.add_argument(
    '-w', '--work',
    help="Add a writable directory to mounts",
    type=Path,
    action='append',
    default=[])

  parser.add_argument(
    '-c', '--cwd',
    help="",
    type=Path,
    default=Path.cwd())

  parser.add_argument(
    '--knowledge',
    help="",
    type=Path,
    default=None)

  # cmd (positional, collects all remaining)
  # Using REMAINDER so everything after this argument is captured "as-is."
  # parser.add_argument(
  #   'cmd',
  #   nargs='*',
  #   help="Command and its arguments")

  # args = parser.parse_args()
  args, extra = parser.parse_known_args()

  command = None

  if extra:
    if extra[0] != '--':
      parser.print_help()
      return 1

    command = extra[1:]


  if args.init:
    image = args.image

    if image is None:
      image = _CONFIG.get('default_image')

    if init(
      image = image,
      basis = args.basis,
      cwd = args.cwd,
      python = args.python):

      return 1

  if command is not None and not command:
      return 0

  return run(
    image = args.image,
    basis = args.basis,
    app = args.app,
    read = args.read,
    work = args.work,
    host_read = args.host_read,
    cwd = args.cwd,
    cmd = command or [],
    device = args.device,
    oci_mode = args.oci_mode,
    refresh_cdi = args.refresh_cdi,
    silent = args.silent,
    network = args.network,
    knowledge = args.knowledge)

#===============================================================================
exit(main())
