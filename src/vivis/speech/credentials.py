"""Local persistent key or systemd credential; never include secrets in errors."""
import os
from pathlib import Path
import stat
import tempfile


def default_path():return Path.home()/'.config/vivis/dashscope_api_key'


def validate(value):
    if not isinstance(value,str) or not 1<=len(value)<=4096 or not value.isascii() or any(c.isspace() or not c.isprintable() for c in value):
        raise ValueError('API Key为空或格式无效；请重新配置，不要将密钥写入日志')
    return value


def load_key(name='DASHSCOPE_API_KEY',environ=None,path=None):
    env=os.environ if environ is None else environ
    if env.get(name):return validate(env[name])
    credential=env.get('CREDENTIALS_DIRECTORY')
    location=Path(credential)/'dashscope_api_key' if credential else Path(path or default_path())
    try:
        fd=os.open(location,os.O_RDONLY|os.O_NOFOLLOW)
    except FileNotFoundError:return ''
    except OSError:raise ValueError('无法读取本机API Key文件，请检查权限') from None
    with os.fdopen(fd) as stream:
        info=os.fstat(stream.fileno())
        mode=stat.S_IMODE(info.st_mode)
        # systemd255 materializes credentials as root:root0440 in a private
        # per-service mount. This differs from the user's persistent0600 file.
        restricted=(mode & ~0o440)==0 if credential else (mode & 0o077)==0
        if not stat.S_ISREG(info.st_mode) or not restricted or info.st_uid not in (0,os.geteuid()):
            raise ValueError('systemd凭据必须为受限只读普通文件' if credential else 'API Key文件必须为普通文件且权限为600或更严格')
        return validate(stream.read(4098).rstrip('\r\n'))


def save_key(value,path=None):
    secret=validate(value);path=Path(path or default_path())
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd,temporary=tempfile.mkstemp(prefix='.api-key-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:stream.write(secret+'\n')
        os.chmod(temporary,0o600);os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path
