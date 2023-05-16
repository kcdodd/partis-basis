CLI utility for using Singularity container as virtual environment (requires singularity to already be installed).

```
$ basis --help
Usage: basis [OPTIONS] [CMD]...

Options:
  -i, --image TEXT
  -a, --app TEXT    {bash, fish, jupyter, code}
  -b, --basis PATH
  -w, --work PATH
  -c, --cwd PATH
  --help            Show this message and exit.
```

The first time it is run it will download a pre-build Singularity image.
You must also tell it where to create the basis directory on the first run, which
will become a virtual home inside the container.

```bash
$ basis -b ./.basis
image: library://kcdodd/basis/ubuntu-22.04-basis:0.0.1
  app: fish
basis: /path/to/cwd/.basis
 work: /
  cwd: /path/to/cwd
  cmd:
Entering container...
INFO:    Using cached image
Loading python/3.11.0/gcc-system
  Loading requirement: gcc/system/default
Creating environment: /basis/envs/default
Creating modulefile /basis/mods/environment/default
Loading environment: /basis/envs/default
Saved modules to /basis/envs/default/modules
gcc/system/default python/3.11.0/gcc-system environment/default
Loading environment: /basis/envs/default
Welcome to fish, the friendly interactive shell
Type help for instructions on how to use fish
>? ok | 10:43:13 | 0ms
>[S] user@hostname | (default) | /work/path/to/cwd
>$ echo $HOME
/basis
>? ok | 10:55:36 | 0ms
>[S] user@hostname | (default) | /work/path/to/cwd
>$ which python
/basis/envs/default/bin/python
>? ok | 10:59:06 | 6ms
>[S] user@hostname | (default) | /work/path/to/cwd
>$ exit
... Exited container
```

Safely delete basis directory to start over.

```
rm -r ./.basis