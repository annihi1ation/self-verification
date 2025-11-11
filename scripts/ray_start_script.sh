cd verl-train

conda activate verl
pip install torch-memory-saver unidiff
pip install liger_kernel
pip install -e .

# sglang backend
# pip install "sglang[all]>=0.4.6.post4"

# vllm backend
pip install vllm==0.8.2 -i https://mirrors.ustc.edu.cn/pypi/simple
pip install tensordict==0.6.2 -i https://mirrors.ustc.edu.cn/pypi/simple

# Check if this node is the master
if [ "$(hostname)" = "${MASTER_ADDR}" ]; then

ray start --head --node-ip-address ${MASTER_ADDR} --port ${MASTER_PORT}

# Can put any rl scripts here to directly submit jobs

else

ray start --address ${MASTER_ADDR}:${MASTER_PORT}

fi

sleep infinity
