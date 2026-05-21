export RLINF_NODE_RANK=0

unset PYTHONPATH
export DREAMZERO_PATH=/home/shiliangzhi/work-space/wyt/deploy_dz/dreamzero
export PYTHONPATH=$DREAMZERO_PATH:$PYTHONPATH:/home/shiliangzhi/work-space/wyt/deploy_dz/RLinf
export RLINF_COMM_NET_DEVICES=rlinf

source /home/shiliangzhi/work-space/wyt/deploy_dz/RLinf/.venv/bin/activate