import os
from dotenv import load_dotenv
load_dotenv()

def get_env(env_name: str, default=None, *, required: bool = True):
    if env_name in os.environ:
        return os.environ[env_name]
    if not required:
        return default
    raise Exception(f'{env_name} does not exist in environment')