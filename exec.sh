mkdir runs
for i in {1,2,3}
do
    python3 project/ppo_agent.py
    mv train_ppo.txt ./runs/train_ppo_${i}.txt
    mv eval_ppo.txt ./runs/eval_ppo_${i}.txt
    python3 project/agent_dqn.py
    mv train_dqn.txt ./runs/train_dqn_${i}.txt
    mv eval_dqn.txt ./runs/eval_dqn_${i}.txt
done
