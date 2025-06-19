# message = "move the cup on top of the box"
# message = "get the remote to person"
# message = "get the mouse to person"
message = "move to the mouse"
message = "grab the mouse"
message = "grab the cup"
message = "put the green ball into the blue basket"
message = "release the greeb ball on top of the cup"
message = "release the greeb ball on top of the cup, please also watch out for the red ball"

# message = "turn on the switch"
# message = "put the red ball into the green basket"
# message = "close the top drawer"
message = "pour water out of the cup"


import numpy as np
from openai import AzureOpenAI
from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from utils import set_lmp_objects
from src.toolbox.real_env import RealRobotEnv
import configparser
import os

objects = dict()
currentpos = dict()

# Load Azure OpenAI config from config.ini
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '../config.ini')
config.read(config_path)
azure_endpoint = config['azure_openai']['azure_endpoint']
api_version = config['azure_openai']['api_version']
api_key = config['azure_openai']['api_key']

client = AzureOpenAI(
  azure_endpoint = azure_endpoint,
  api_version = api_version,
  api_key = api_key
)




def gpt(message, instruction):
    response = client.chat.completions.create(
		model = 'gpt-4',
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": message}
        ]
    )
    print(response.usage)
    return response.choices[0].message.content


T_camera_to_base = np.array([
    [  0.771194,  -0.462891,   0.437026, -455.106040],
    [ -0.636553,  -0.552384,   0.538212, -947.105010],
    [ -0.007728,  -0.693256,  -0.720650, 578.839264],
    [  0.000000,   0.000000,   0.000000,   1.000000]
])


config = get_config('rlbench')
visualizer = ValueMapVisualizer(config['visualizer'])
env = RealRobotEnv(serial_port, 
                   visualizer=visualizer, 
                   instruction=message, 
                   T_camera_to_base = T_camera_to_base, 
                   debug=False)
lmps, lmp_env = setup_LMP(env, config, debug=False)
voxposer_ui = lmps['plan_ui']


objects_on_table = env.objects_on_table
print('[INFO]: Objects on table:', objects_on_table)
instruction = open('src/toolbox/my_prompt/get_target.txt').read()
response = gpt(message, instruction)
target_objects = eval(response)
print('[INFO]: Target objects:', target_objects)

env.load_task(target_objects)
set_lmp_objects(lmps, target_objects)

voxposer_ui(message)

env.__del__()