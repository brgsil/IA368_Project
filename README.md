# Projeto Final IA368 - Aprendizado por Reforço

O cenário de aprendizado contínuo com aprendizado por reforço busca desenvolver métodos para o treinamento de agentes em MDP não-estacionários. Em especial, MDP com conjuntos discretos de dinâmicas de troca de estados e recompensas, denominados tarefas, apresentam uma família de problemas menos complexos para abordar o cenário de aprendizado contínuo. Dentro desse cenário, esse trabalho busca comparar o desempenho de um método de *pseudo-rehearsal* (RePR) contra o algoritmo PPO-Clip. Para isso, é utilizado diferentes instâncias do ambiente Lunar Lander inicializadas com diferentes parâmetros. Os agentes são treinados de forma sequecial em cinco configurações distintas do ambiente e o desempenho é avaliado para a tarefa (ambiente) atual assim como as anteriores. Um agente com boa capacidade de aprendizado contínuo consegue obter bom desempenho na tarefa atual sem deteriorar seu desempenho nas tarefas anteriores. A implementação é feita com o framework Pytorch e baseada em implementação dos métodos disponíveis em repositórios públicos. O modelo PPO-Clip apresentou ótimo desempenho, sendo capaz de adaptar o conhecimento adquirido em tarefas anteriores para novas tarefas, enquanto o DDQN do RePR apresentou um treinamento menos estável.

## Pré-requisito

A execução do experimento é feita a partir de um container Docker.
Para isso, é necessária realizar a instalação do Docker em sua máquina.
Pode-se seguir os tutoriais presentes no site do [Docker](https://docs.docker.com/engine/install/)

## Execução

1. Criação da imagem Docker.
A partir do diretório principal do projeto, execute:
```
docker build . -t projeto-repr
```

2. Execução do container
```
docker run --rm -it projeto-repr
```

3. Execução do script de experimentos. Após inicializar o container Docker, o terminal do mesmo irá aparecer em seu terminal, basta então executar o script:
```
./exec.sh
```
Após a execução dos experimentos, os resultados são salvos no diretório ```/runs``` e os gráficos podem ser gerados por ```python3 plot.py```.
