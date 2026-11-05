# Modulo-3_Machine_learn

## 1. Entendimento do problema e coleta.
- Problema: Identificar o nivel de satisfação e insatisfação de acordo com os serviços prestados.
- Publico-alvo: Jovens adultos e Adultos
- Decisão: tipos errados e alguns numeros nulos em Arrival Delay in Minutes.

## fonte:
- O conjunto de dados Airline Passenger Satisfaction,disponibilizado na plataforma Kaggle por Atchireddypammi, reúne informações relacionadas a satisfação de passageiros em voos de diferentes horarios, idade, etc.utilizando Python e Pandas para carregar os arquivos train.csv e test.csv, permitindo sua análise. Esse dataset pode ser utilizado para explorar analise de perfis, melhorias de atendimento e padrões de insatisfação.

## 2. Conclusão:
- Após a realização das etapas de pré-processamento e limpeza dos dados, foi aplicado o método SMOTE e UnderSample para equilibrar as classes da base de treinamento, reduzindo possíveis impactos do desbalanceamento na aprendizagem do modelo.

- Em seguida, foi treinado o modelo Random Forest, configurado com 50 árvores de decisão e uma semente aleatória fixa para garantir a reprodutibilidade dos resultados. Após o treinamento, o modelo foi submetido a uma avaliação utilizando uma base de teste independente (datatest.csv), permitindo verificar sua capacidade de generalização diante de dados que não participaram do processo de aprendizagem.

- Foram utilizadas métricas como acurácia, precisão ponderada, recall e F1-score, por meio do relatório de classificação, com o objetivo de analisar o desempenho do modelo na identificação de passageiros satisfeitos e neutros ou insatisfeitos.

- Dessa forma, o desenvolvimento do modelo representa uma etapa importante na construção de uma solução de aprendizado de máquina voltada à predição da satisfação dos passageiros, permitindo identificar padrões associados à experiência dos clientes e fornecendo subsídios para futuras estratégias de melhoria da qualidade dos serviços.
