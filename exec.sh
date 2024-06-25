mkdir runs
for i in {1,2,3,4,5}
do
    python project/ppo_agent.py
    mv train_ppo.txt ./runs/train_ppo_${i}.txt
    mv eval_ppo.txt ./runs/eval_ppo_${i}.txt
    python project/agent_dqn.py
    mv train_dqn.txt ./runs/train_dqn_${i}.txt
    mv eval_dqn.txt ./runs/eval_dqn_${i}.txt
done
