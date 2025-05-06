import os
from dotenv import load_dotenv
load_dotenv()

def get_env(env_name: str):
    if env_name not in os.environ:
        raise Exception(f'{env_name} does not exist in environment')
    return os.environ[env_name]