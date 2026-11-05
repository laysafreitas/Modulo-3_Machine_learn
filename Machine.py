import pandas as pd
import numpy as np
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
import statsmodels.api as sm

RANDOM_STATE = 42
TIPO = "classificacao"

df = pd.read_csv("C:/Users/Aluno/Downloads/Analise de dados-20261027T182755Z-1-001/LLM/airline_passenger_satisfaction.csv")

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

df_train.to_csv("dados_train.csv", index=False)
df_test.to_csv("dados_test.csv", index=False)

print(
    f"Linhas no treino: {df_train.shape[0]} | Linhas no teste:"
    f" {df_test.shape[0]}"
)

X = df_train.drop('satisfaction', axis=1)
Y = df_train['satisfaction']

X_train, X_test, y_train, y_test = train_test_split(
        X,Y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=Y if TIPO == "classificacao" else None
    )

print(f"Treino: {X_train.shape[0]} linhas | Teste: {X_test.shape[0]} linhas")

# 3. Experimento Inicial: Árvore de Decisão

contagem_classes = Y.value_counts()
proporcao_classes = Y.value_counts(normalize=True) * 100
print("Distribuição original da variável alvo (satisfaction):")
for cls, count in contagem_classes.items():
    print(f"Classe {cls}: {count} amostras ({proporcao_classes[cls]:.2f}%)")
    
    
n_iter = 30
test_size = 0.2
np.random.seed(19)

DT = DecisionTreeClassifier(max_depth=3, random_state=19)

acuracia_arvore = []

for i in range(n_iter):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=test_size)

    DT.fit(X_train, Y_train)
    DT_pred = DT.predict(X_test)

    acuracia = accuracy_score(y_test, DT_pred)
    acuracia_arvore.append(acuracia)
    
media_acuracia = np.mean(acuracia_arvore)
desvio_padrao_acuracia = np.std(acuracia_arvore)


print(f"Média da acurácia (Árvore de Decisão): {media_acuracia:.4f}")


print(f"Desvio padrão da acurácia: {desvio_padrao_acuracia:.4f}")

print("Acurácias registradas por iteração:")

print([round(acc, 4) for acc in acuracia_arvore])

matriz_confusao = confusion_matrix(y_test, DT_pred)
relatorio_classificacao = classification_report(y_test, DT_pred, target_names=['Desocupado (0)', 'Ocupado (1)'])

print("Matriz de Confusão (Última Iteração):\n", matriz_confusao)

print("\nRelatório de Classificação Detalhado:\n", relatorio_classificacao)


# 4. Undersampling (RandomUnderSampler)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


undersampler_demo = RandomUnderSampler(random_state=0)
X_demo_res, y_demo_res = undersampler_demo.fit_resample(X, Y)
print("Contagem de classes após o Undersampling na base total:", sorted(Counter(y_demo_res).items()))


acuracia_arvore_U, precisao_arvore_U = [], []
acuracia_rf_U, precisao_rf_U = [], []

undersampler = RandomUnderSampler(random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(X, Y), 1):
    # Separação dos dados de treino e validação da dobra corrente
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Aplicação do Undersampling exclusivamente nos dados de TREINO (evitando data leakage)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

    # 1. Árvore de Decisão
    modelo_arvore = DecisionTreeClassifier(random_state=42)
    modelo_arvore.fit(X_resampled, y_resampled)
    pred_arvore = modelo_arvore.predict(X_test)
    acuracia_arvore_U.append(accuracy_score(y_test, pred_arvore))
    precisao_arvore_U.append(precision_score(y_test, pred_arvore, average='weighted'))

    # 4. Random Forest
    modelo_rf = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_rf.fit(X_resampled, y_resampled)
    pred_rf = modelo_rf.predict(X_test)
    acuracia_rf_U.append(accuracy_score(y_test, pred_rf))
    precisao_rf_U.append(precision_score(y_test, pred_rf, average='weighted'))

print("Validação cruzada com Undersampling concluída com sucesso.")

# Oversampling (SMOTE)

# Listas para armazenamento das métricas por fold - Oversampling
acuracia_arvore_O, precisao_arvore_O = [], []
acuracia_rf_O, precisao_rf_O = [], []

# Instanciando a técnica SMOTE para sobreamostragem sintética
oversampler = SMOTE(random_state=42)

# Loop de Validação Cruzada (5 Folds)
for fold, (train_index, test_index) in enumerate(kf.split(X, Y), 1):
    # Separação dos dados da dobra
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Aplicação do SMOTE estritamente no conjunto de TREINO da dobra
    X_resampled_O, y_resampled_O = oversampler.fit_resample(X_train, y_train)

    # 1. Árvore de Decisão
    modelo_arvore_O = DecisionTreeClassifier(random_state=42)
    modelo_arvore_O.fit(X_resampled_O, y_resampled_O)
    pred_arvore_O = modelo_arvore_O.predict(X_test)
    acuracia_arvore_O.append(accuracy_score(y_test, pred_arvore_O))
    precisao_arvore_O.append(precision_score(y_test, pred_arvore_O, average='weighted'))

    # 4. Random Forest
    modelo_rf_O = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_rf_O.fit(X_resampled_O, y_resampled_O)
    pred_rf_O = modelo_rf_O.predict(X_test)
    acuracia_rf_O.append(accuracy_score(y_test, pred_rf_O))
    precisao_rf_O.append(precision_score(y_test, pred_rf_O, average='weighted'))

print("Validação cruzada com Oversampling (SMOTE) concluída com sucesso.")

#  Análise Comparativa e Consolidação dos Resultados

# Construção de uma tabela resumo para comparação direta das médias obtidas na validação cruzada
df_resultados = pd.DataFrame({
    'Técnica': ['Undersampling', 'Undersampling',
                'Oversampling (SMOTE)','Oversampling (SMOTE)'],
    'Modelo': ['Árvore de Decisão', 'Random Forest',
               'Árvore de Decisão', 'Random Forest'],
    'Acurácia Média': [
        np.mean(acuracia_arvore_U), np.mean(acuracia_rf_U),
        np.mean(acuracia_arvore_O), np.mean(acuracia_rf_O)
    ],
    'Precisão Média': [
        np.mean(precisao_arvore_U), np.mean(precisao_rf_U),
        np.mean(precisao_arvore_O), np.mean(precisao_rf_O)
    ]
})

# Exibição organizada dos resultados ordenados pela melhor Acurácia Média
print(df_resultados.sort_values(by='Acurácia Média', ascending=False).reset_index(drop=True))

# Treinamento Final no Dataset Completo

# 1. Aplicação do SMOTE na totalidade da base de treino limpa
smote_final = SMOTE(random_state=42)
X_final_res, y_final_res = smote_final.fit_resample(X, Y)

# 2. Treinamento do modelo campeão (Random Forest) com todos os dados balanceados
modelo_campeao = RandomForestClassifier(n_estimators=50, random_state=42)
modelo_campeao.fit(X_final_res, y_final_res)

X_teste_indep = df_test.drop('satisfaction', axis=1)
y_teste_indep = df_test['satisfaction']

pred_final = modelo_campeao.predict(X_teste_indep)
print("=== AVALIAÇÃO NO DATASET DE TESTE INDEPENDENTE (datatest.csv) ===")
print(f"Acurácia Final: {accuracy_score(y_teste_indep, pred_final):.4f}")
print(f"Precisão Final: {precision_score(y_teste_indep, pred_final, average='weighted'):.4f}")
print("\nRelatório de Classificação Final:\n", classification_report(y_teste_indep, pred_final, target_names=['neutro ou insatisfeito', 'satisfeito']))

# 4. Salvando o modelo treinado em disco utilizando joblib
nome_arquivo_modelo = 'modelo_occupancy_rf.joblib'
joblib.dump(modelo_campeao, nome_arquivo_modelo)
print(f"Modelo salvo com sucesso no arquivo: '{nome_arquivo_modelo}'")