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
import pwd
from subprocess import check_output, check_call, CalledProcessError, STDOUT

#===============================================================================
_PY = f"{'.'.join(str(v) for v in sys.version_info[:2])}"
_CONFIG_DIR = Path.home()/'.config'/'partis-basis'
_CONFIG_FILE = _CONFIG_DIR/'config.json'

#===============================================================================
# directory for Container Device Interface (CDI) configs
_CDI_DIR = _CONFIG_DIR/'cdi'
# NOTE: the CDI default search directories, managed by the host (root). Only
# specs in _CDI_DIR are generated/updated here.
_CDI_DIR_HOST = [Path('/etc/cdi'), Path('/var/run/cdi')]
# extensions the runtime recognizes as a spec
_CDI_SUFFIX = ('.yaml', '.json')
_CDI_NVIDIA = 'nvidia.yaml'
_CDI_NVIDIA_KIND = 'nvidia.com/gpu'

# 'kind: nvidia.com/gpu', anchored so a nested value cannot match
_CDI_KIND_REC = re.compile(
  r'^\s*"?kind"?\s*:\s*"?(?P<kind>[\w.\-]+/[\w.\-]+)"?',
  re.MULTILINE)

# device names ('  name: "0"'); the deeper 'hookName' keys cannot match
_CDI_NAME_REC = re.compile(
  r'^\s*-?\s*"?name"?\s*:\s*"?(?P<name>[\w.\-:]+)"?',
  re.MULTILINE)

#===============================================================================
# The image this version runs, unless '--image' names another one. Not
# configurable and not recorded in the basis directory: an environment runs
# whichever image the installed partis-basis names, so install an older
# partis-basis to go back to an older image.
_IMAGE = "oras://ghcr.io/kcdodd/partis-basis/ubuntu-22.04-basis:0.3.0"

#===============================================================================
_APPTAINER_INSTALL = "https://apptainer.org/docs/admin/main/installation.html"
# first apptainer release with native CDI devices ('--device', '--cdi-dirs')
_APPTAINER_CDI = (1, 5)

# 'apptainer version 1.5.5', 'singularity-ce version 4.6.0'
_VERSION_REC = re.compile(
  r'^(?P<name>apptainer|singularity)\S*\s+version\s+(?P<major>\d+)\.(?P<minor>\d+)')

#===============================================================================
_CONFIG = {}
_CONFIG_DEFAULTS = {
  # 'auto' -> generate nvidia CDI spec on demand, 'off' -> never generate
  "nvidia_cdi": "auto"}

if _CONFIG_FILE.exists():
  try:
    _CONFIG = json.loads(_CONFIG_FILE.read_bytes())

  except (OSError, ValueError) as e:
    print(
      f"Warning: Ignoring unreadable config {_CONFIG_FILE}: {e}",
      file=sys.stderr)

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
  # NOTE: a scheduler sets this for a gpu allocation (e.g. slurm GRES), so it is
  # inherited when '--gpu' does not select devices itself
  'CUDA_VISIBLE_DEVICES',
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
def _err(*lines: str):
  """Report to stderr, so error text does not land in piped output"""
  for line in lines:
    print(line, file=sys.stderr)

#===============================================================================
def _tail(output: bytes|None, n: int = 5) -> list[str]:
  lines = (output or b'').decode('utf-8', errors='replace').splitlines()

  return [line for line in lines if line.strip()][-n:]

#===============================================================================
def _csv(values: list[str]) -> list[str]:
  """Flatten repeated and comma separated option values, preserving order"""
  names = []

  for value in values:
    for name in value.split(','):
      name = name.strip()

      if name and name not in names:
        names.append(name)

  return names

#===============================================================================
def _gpu_names(values: list[str]) -> list[str]:
  """Normalize a '--gpu' selection to CDI device names"""
  names = _csv(values)

  if 'all' in names:
    # 'all' subsumes any other selection, and injecting one device twice is a
    # CDI conflict rather than a duplicate
    return ['all']

  return names

#===============================================================================
def _subid(path: Path = Path('/etc/subuid')) -> bool:
  """Whether a subordinate uid range exists for the current user

  OCI mode maps ids through this range; the file is often present but empty.
  """
  uid = os.getuid()
  names = {str(uid)}

  try:
    names.add(pwd.getpwuid(uid).pw_name)
  except KeyError:
    pass

  try:
    return any(
      line.split(':')[0] in names
      for line in path.read_text().splitlines())

  except OSError:
    return False

#===============================================================================
def _oci_deps():
  """Report OCI mode dependencies that are missing

  Advisory only: singularity is the authority on what its OCI mode requires, but
  the failure it reports for a missing one does not name it.
  """
  missing = []

  if not (shutil.which('crun') or shutil.which('runc')):
    missing.append("'crun' or 'runc' (OCI runtime)")

  for name, why in [
      ('conmon', 'container monitor'),
      ('fuse-overlayfs', 'writable overlay'),
      ('newuidmap', 'user namespace id mapping')]:

    if not shutil.which(name):
      missing.append(f"{name!r} ({why})")

  if os.getuid() != 0 and not _subid():
    missing.append("a '/etc/subuid' range for the current user")

  if missing:
    _err("Warning: OCI mode dependencies not found:")

    for line in missing:
      _err(f"  {line}")

#===============================================================================
def _nvidia_driver() -> str|None:
  """Version of the host nvidia kernel driver, or None if no driver is present

  Tests for the driver itself, not the container toolkit. 'libnvidia-ml.so.1'
  is exactly what 'nvidia-ctk' needs to enumerate devices.
  """
  try:
    nvml = ctypes.CDLL('libnvidia-ml.so.1')
  except OSError:
    return None

  # native device nodes, or WSL2 (driver libraries under '/usr/lib/wsl/lib')
  if not (Path('/dev/nvidiactl').exists() or Path('/dev/dxg').exists()):
    return None

  version = Path('/sys/module/nvidia/version')

  try:
    if version.is_file():
      return version.read_text().strip() or 'unknown'

  except OSError:
    pass

  # '/sys/module/nvidia' is absent on WSL2. NVML reports the same version there,
  # but costs an 'nvmlInit', so it is only the fallback.
  return _nvidia_driver_nvml(nvml) or 'unknown'

#===============================================================================
def _nvidia_driver_nvml(nvml: ctypes.CDLL) -> str|None:
  buf = ctypes.create_string_buffer(80)

  try:
    if nvml.nvmlInit_v2() != 0:
      return None

    try:
      if nvml.nvmlSystemGetDriverVersion(buf, len(buf)) != 0:
        return None

    finally:
      nvml.nvmlShutdown()

  except (AttributeError, OSError):
    return None

  return buf.value.decode('utf-8', errors='replace').strip() or None

#===============================================================================
def _cdi_spec(spec: Path) -> tuple[str|None, list[str]]:
  """The device kind a CDI spec declares, and the device names it declares

  Parsed only well enough to tell specs apart and to check that a requested
  device exists. Returns (None, []) for anything not understood: the runtime
  remains the authority, this only gives a better error before reaching it.
  """
  try:
    text = spec.read_text(errors='replace')

  except OSError:
    return None, []

  if spec.suffix == '.json':
    try:
      data = json.loads(text)

      return (
        data.get('kind'),
        [
          name
          for device in (data.get('devices') or [])
          if isinstance(name := device.get('name'), str)])

    except (ValueError, AttributeError):
      return None, []

  kind = _CDI_KIND_REC.search(text)

  # NOTE: 'devices' precedes 'kind' in a generated spec, so the whole file has
  # to be read either way
  return (
    kind.group('kind') if kind else None,
    _CDI_NAME_REC.findall(text))

#===============================================================================
def _cdi_host_spec(kind: str) -> Path|None:
  """Host managed spec declaring 'kind', if there is one

  Detected by content rather than filename: host tooling chooses the filename,
  and a host spec missed here means a second spec declaring the same devices,
  which the runtime resolves as a conflict.
  """
  for _dir in _CDI_DIR_HOST:
    try:
      specs = sorted(p for p in _dir.iterdir() if p.suffix in _CDI_SUFFIX)

    except OSError:
      continue

    for spec in specs:
      if _cdi_spec(spec)[0] == kind:
        return spec

  return None

#===============================================================================
def _cdi_dirs() -> list[Path]:
  """CDI search directories that exist, most specific first

  A search directory that does not exist is not skipped by singularity, and only
  _CDI_DIR can be created here. '/var/run' is a symlink to '/run' on most hosts,
  so resolve before de-duplicating: one spec reached by two paths declares its
  devices twice.
  """
  dirs = []

  for _dir in [_CDI_DIR, *_CDI_DIR_HOST]:
    try:
      if not _dir.is_dir():
        continue

      _dir = _dir.resolve()

    except OSError:
      continue

    if _dir not in dirs:
      dirs.append(_dir)

  return dirs

#===============================================================================
def _nvidia_cdi(names: list[str], *, refresh: bool = False) -> int:
  """Ensure a CDI spec declaring the requested nvidia gpu devices is available

  Returns non-zero if a requested device cannot be made available.
  """
  if _CONFIG.get('nvidia_cdi') == 'off':
    # host is assumed to manage CDI specs entirely
    if refresh or _debug():
      print(
        f"Not generating CDI config for {_CDI_NVIDIA_KIND}:"
        f" 'nvidia_cdi' is 'off' in {_CONFIG_FILE}")

    return 0

  cdi = _CDI_DIR/_CDI_NVIDIA
  # a generated spec references version suffixed driver libraries, so it has to
  # be re-generated after a host driver upgrade
  stamp = _CDI_DIR/f'{_CDI_NVIDIA}.driver'
  host = _cdi_host_spec(_CDI_NVIDIA_KIND)

  if host is not None:
    # The host managed spec wins, but _CDI_DIR is searched first: a generated
    # spec left in place would declare the same devices as the host one.
    try:
      cdi.unlink(missing_ok=True)
      stamp.unlink(missing_ok=True)

    except OSError as e:
      _err(f"Error: Could not remove generated CDI config: {e}")
      return 1

    if refresh:
      print(f"Host managed CDI config, not re-generated: {host}")

    return _nvidia_cdi_check(host, names)

  driver = _nvidia_driver()

  if driver is None:
    _err(
      f"Error: {_CDI_NVIDIA_KIND} devices require an nvidia driver"
      " ('libnvidia-ml.so.1' not found)")
    return 1

  nvidia_ctk = shutil.which('nvidia-ctk')

  if nvidia_ctk is None:
    _err(
      f"Error: {_CDI_NVIDIA_KIND} devices require 'nvidia-ctk'"
      " (nvidia-container-toolkit), or a host managed CDI config in"
      f" {', '.join(str(v) for v in _CDI_DIR_HOST)}")
    return 1

  try:
    stale = not (
      cdi.is_file()
      and stamp.is_file()
      and stamp.read_text().strip() == driver)

  except OSError as e:
    _err(f"Error: Could not read CDI driver stamp {stamp}: {e}")
    return 1

  if (stale or refresh) and _nvidia_cdi_generate(nvidia_ctk, driver, cdi, stamp):
    return 1

  return _nvidia_cdi_check(cdi, names)

#===============================================================================
def _nvidia_cdi_generate(
    nvidia_ctk: str,
    driver: str,
    cdi: Path,
    stamp: Path) -> int:
  """Generate the nvidia CDI spec in _CDI_DIR"""
  cmd = [nvidia_ctk, 'cdi', 'generate', '--format=yaml']
  tmp = None

  try:
    _CDI_DIR.mkdir(exist_ok=True, parents=True)

    # Generate out of place so a failed run cannot leave a truncated spec, but
    # not within _CDI_DIR itself, since anything there with a spec extension is
    # a spec as far as the runtime is concerned.
    fd, _tmp = tempfile.mkstemp(
      dir=_CDI_DIR.parent,
      prefix=f'.{_CDI_NVIDIA}.',
      suffix='.yaml')

    os.close(fd)
    tmp = Path(_tmp)
    cmd.append(f'--output={tmp}')

    # nvidia-ctk is chatty on stderr even when it succeeds
    output = check_output(cmd, stderr=STDOUT)

    if not tmp.stat().st_size:
      raise OSError(f"nothing written to {tmp}")

    os.replace(tmp, cdi)
    # installed: no longer a temporary to clean up
    tmp = None

    stamp.write_text(driver)

  except (CalledProcessError, OSError) as e:
    _err(
      "Error: Could not create nvidia Container Device Interface (CDI)"
      f" config: {e}")

    for line in _tail(getattr(e, 'output', None)):
      _err(f"  {line}")

    _err(f"  command: {shlex.join(cmd)}")
    return 1

  finally:
    if tmp is not None:
      # NOTE: nvidia-ctk appends '.yaml' to the output path if the extension is
      # not one it recognizes, which would otherwise leak a spec sized file
      for path in [tmp, Path(f'{tmp}.yaml')]:
        try:
          path.unlink(missing_ok=True)

        except OSError:
          pass

  if _debug():
    print(f"Generated CDI config for nvidia driver {driver}: {cdi}")

    for line in _tail(output, 100):
      print(f"  {line}")

  return 0

#===============================================================================
def _nvidia_cdi_check(spec: Path, names: list[str]) -> int:
  """Verify a CDI spec declares each requested device"""
  available = _cdi_spec(spec)[1]

  if not available:
    # nothing parsed: leave the runtime as the authority rather than guess
    if _debug():
      print(f"No device names parsed from CDI config: {spec}")

    return 0

  missing = [name for name in names if name not in available]

  if missing:
    _err(
      f"Error: CDI config declares no {_CDI_NVIDIA_KIND} device"
      f" {', '.join(repr(v) for v in missing)}: {spec}",
      f"  available: {', '.join(available)}")
    return 1

  return 0

#===============================================================================
def _runtime() -> tuple[str, str, tuple[int, int]]|None:
  """The container runtime: apptainer, or singularity as a deprecated fallback

  Returns (executable, name, (major, minor)), or None if neither is usable.
  Identified by its version output rather than the program name, since
  apptainer installs a 'singularity' link to itself.
  """
  for prog in ['apptainer', 'singularity']:
    exe = shutil.which(prog)

    if not exe:
      continue

    try:
      output = check_output([exe, '--version'], stderr=STDOUT)
    except (CalledProcessError, OSError):
      continue

    m = _VERSION_REC.match(output.decode('utf-8', errors='replace').strip())

    if not m:
      _err(f"Warning: Unrecognized version from {exe}: {_tail(output, 1)}")
      continue

    name = m.group('name')

    if name == 'singularity':
      _err(
        "Warning: Using singularity, which is deprecated for partis-basis and"
        " will not be supported in a future release. Install apptainer:",
        f"  {_APPTAINER_INSTALL}")

    return exe, name, (int(m.group('major')), int(m.group('minor')))

  _err(
    "Error: Program 'apptainer' not found. Install apptainer:",
    f"  {_APPTAINER_INSTALL}")

  return None

#===============================================================================
def _basis(*, basis: Path|None = None) -> tuple[Path, Path]:
  """The basis directory to bind into the container, created if it is not there"""

  if basis is None:
    venv: str|Path = os.environ.get('VIRTUAL_ENV', '')

    if venv:
      venv = Path(venv).resolve()

    if not (venv and venv.exists()):
      raise FileNotFoundError(f"No active virtual environment: {venv}")

    basis = venv/'.basis'

  default = basis/'mods'/'environment'/'.version'

  if not basis.exists():
    basis.mkdir(parents=True)

  return basis, default

#===============================================================================
def init(*, image, basis, cwd, python):
  name = 'init'
  # NOTE: not 'pwd', which is a stdlib module used here
  cwd_ = cwd.resolve()

  runtime = _runtime()

  if runtime is None:
    return 1

  exe = runtime[0]

  basis, default = _basis(basis=basis)
  env_dir = basis/'envs'/name

  if env_dir.is_dir() and any(env_dir.iterdir()):
    _err(f"Cannot initialize non-empty directory: {env_dir}")
    return 1

  env = {
    k:v
    for k,v in os.environ.items()
    if k in PASS_ENV or MPI_ENV_REC.fullmatch(k)}

  def _run(*cmd):
    check_call([
      exe,
      'run',
      '--app', 'manage',
      '--no-home',
      '--pwd', str(cwd_),
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
    gpu,
    device,
    oci_mode,
    refresh_cdi,
    silent,
    network,
    knowledge,
    claude_md):
  runtime = _runtime()

  if runtime is None:
    return 1

  exe, runtime_name, runtime_version = runtime
  apptainer = runtime_name == 'apptainer'

  if oci_mode and apptainer:
    _err("Error: --oci requires singularity; apptainer has no OCI mode")
    return 1

  if device and apptainer and runtime_version < _APPTAINER_CDI:
    _err(
      "Error: --device requires apptainer"
      f" {'.'.join(str(v) for v in _APPTAINER_CDI)} or later (found"
      f" {'.'.join(str(v) for v in runtime_version)})")
    return 1

  if claude_md:
    claude_md = claude_md.resolve()

    # binding over an existing file in the image requires a file source
    if not claude_md.is_file():
      _err(f"Error: --claude-md is not a file: {claude_md}")
      return 1

  cwd_ = cwd.resolve()
  work += [cwd_]

  basis, default = _basis(basis=basis)

  env = {
    k:v
    for k,v in os.environ.items()
    if k in PASS_ENV or MPI_ENV_REC.fullmatch(k)}

  extra_args = []

  # normalized here as well as in main(): injecting one device twice is a CDI
  # conflict, so this must not depend on the caller having done it
  gpu = _gpu_names(gpu)
  device = _csv(device)

  # singularity supports CDI devices only in OCI mode, so an explicit '--device'
  # requires it there; apptainer supports them natively. Note this is decided
  # before '--gpu' is resolved, which chooses its mechanism from the result.
  oci_mode = oci_mode or (bool(device) and not apptainer)

  if gpu:
    if oci_mode:
      names = gpu

      if names == ['all']:
        # A scheduler allocated subset still applies (e.g. slurm sets
        # 'CUDA_VISIBLE_DEVICES' for a GRES allocation). CDI device names are
        # host indices/uuids, so the selection maps across directly.
        names = _gpu_names([os.environ.get('CUDA_VISIBLE_DEVICES', '')]) or names

      # CDI injects exactly the requested devices rather than filtering, so a
      # host index means nothing inside the container
      env.pop('CUDA_VISIBLE_DEVICES', None)

      # de-duplicated against '--device': the same device named by both routes is
      # still one device
      device = _csv(device + [f'{_CDI_NVIDIA_KIND}={name}' for name in names])

    else:
      if _nvidia_driver() is None:
        _err(
          "Error: --gpu requires an nvidia driver"
          " ('libnvidia-ml.so.1' not found)")
        return 1

      extra_args.append('--nv')

      if refresh_cdi:
        _err("Warning: --refresh-cdi has no effect outside OCI mode, which is"
          " the only mode using CDI")

      if gpu != ['all']:
        # '--nv' binds every host gpu into the container, so this is what
        # actually restricts the selection
        env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu)

  nvidia = [
    name.partition('=')[2]
    for name in device
    if name.startswith(f'{_CDI_NVIDIA_KIND}=')]

  if nvidia and _nvidia_cdi(nvidia, refresh=refresh_cdi):
    return 1

  for name in device:
    extra_args.extend(['--device', name])

  if oci_mode:
    _oci_deps()

    extra_args.extend([
      '--oci',
      '--no-compat'])

  if oci_mode or device:
    # NOTE: only the directories that exist, and only if any do; passing none
    # leaves the runtime with its own defaults
    cdi_dirs = _cdi_dirs()

    if cdi_dirs:
      extra_args.extend([
        '--cdi-dirs', ','.join(str(v) for v in cdi_dirs)])

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

  if claude_md:
    # replaces only the managed CLAUDE.md, not managed-settings.json
    mounts.extend([
      # '--bind', f"{claude_md}:/etc/claude-code/CLAUDE.md:ro"
      '--mount',
      f'type=bind,"src={claude_md}",dst=/etc/claude-code/CLAUDE.md,ro'])

  if host_read:
    mounts.extend([
      '--mount', 'type=bind,src=/,dst=/host,ro'])

  cmd_runtime = [
    exe,
    'run',
    '--app',
    app,
    *extra_args,
    '--no-home',
    '--pwd', str(cwd_),
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

    if claude_md:
      print(f"claude: {repr(str(claude_md))} -> '/etc/claude-code/CLAUDE.md'")
    print(f"  cwd: {repr(str(cwd))}")

    if gpu:
      print(f"  gpu: {', '.join(gpu)} ({'cdi' if oci_mode else '--nv'})")

    if cmd:
      print(f" exec: {shlex.join(cmd)}")

    print(f"Running: {runtime_name} {shlex.join(cmd_runtime[1:])}")

    # flush stdout before replacing process
    sys.stdout.flush()
    sys.stderr.flush()

  os.execve(cmd_runtime[0], cmd_runtime, env)

#===============================================================================
def main():

  parser = argparse.ArgumentParser(
    description="partis-basis runner for apptainer container based environments"
      " (singularity is a deprecated fallback)")

  parser.add_argument(
    '-i', '--image',
    help="Container image to run (default: %(default)s)",
    type=str,
    default=_IMAGE)

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
    help="Run container in OCI mode (singularity only)")

  parser.add_argument(
    '--refresh-cdi',
    dest='refresh_cdi',
    action='store_true',
    help="Re-generate the Container Device Interface (CDI) config for nvidia"
      " gpus. Without --gpu or --device, re-generates and exits")

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
    '--gpu',
    metavar='DEVICES',
    nargs='?',
    const='all',
    action='append',
    default=[],
    help="Make nvidia gpus available: 'all' (the default), or a comma separated"
      " selection of host indices/uuids. In OCI mode the selection is injected"
      " as CDI devices, otherwise '--nv' is used and the selection is applied"
      " with 'CUDA_VISIBLE_DEVICES'. May be repeated")

  parser.add_argument(
    '--device',
    metavar='CDI_DEVICE',
    action='append',
    default=[],
    help="Add a fully qualified CDI device (e.g. 'nvidia.com/gpu=0'). Requires"
      " apptainer >= 1.5, or implies --oci with singularity. May be repeated or"
      " comma separated")

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

  parser.add_argument(
    '--claude-md',
    dest='claude_md',
    metavar='FILE',
    help="Mount a host file (read-only) over the image's"
      " /etc/claude-code/CLAUDE.md. managed-settings.json is not affected",
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

  gpu = _gpu_names(args.gpu)
  device = _csv(args.device)

  if args.refresh_cdi and not (gpu or device):
    # nothing requested to run: re-generate the config and report
    return _nvidia_cdi([], refresh=True)

  if args.init:
    if init(
      image = args.image,
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
    gpu = gpu,
    device = device,
    oci_mode = args.oci_mode,
    refresh_cdi = args.refresh_cdi,
    silent = args.silent,
    network = args.network,
    knowledge = args.knowledge,
    claude_md = args.claude_md)

#===============================================================================
exit(main())
