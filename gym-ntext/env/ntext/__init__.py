
from gym.envs.registration import register


register(
    id='NText-v0',
    entry_point='ntext.envs:NtextEnv',
)

register(
    id='NText-trendy-v0',
    entry_point='ntext.envs:NtextTrendyEnv'
)