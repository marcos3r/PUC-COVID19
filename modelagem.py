"""
# Transformação de dados

Código em Python para análise de dados. 

Este notebook foi desenvolvido para o ambiente GOOGLE COLAB ([colab.research.google.com](https://colab.research.google.com)).

# Inicialização da plataforma

A célula a seguir inicializa a plataforma, carregando as bibliotecas que serão relevantes para o trabalho em seguida.

## Bibliotecas

numpy: usada para processamento numérico.

pandas: usada para manipulação de bases de dados.

pyplot: usada para visualização de dados.

seaborn: usada para visualização de dados.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


 
"""
Parâmetros de configuração para visualização dos dados
"""
np.set_printoptions(threshold=None, precision=2)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('precision', 2)

"""# Base de dados

Essa base de dados pode ser obtida no Kaggle, no endereço: 
https://www.kaggle.com/datasets/einsteindata4u/covid19

"""

from google.colab import files

uploaded = files.upload()

data_set = pd.read_excel(next(iter(uploaded.keys())))

data_positivo = data_set[data_set['SARS-Cov-2 exam result']=='positive']
data_negativo = data_set[data_set['SARS-Cov-2 exam result']=='negative']

data_set = data_set.drop(['Patient addmited to regular ward (1=yes, 0=no)', 
                  'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                  'Patient addmited to intensive care unit (1=yes, 0=no)'],axis=1)
data_set = data_set.set_index('Patient ID')

nan_analyze = data_set.isna().sum()/len(data_set)

print('Quantidade de tuplas:', len(data_set))
print('\nDimensões do DataSet:\n{0}\n'.format(data_set.shape))
print('\nCampos do DataSet:\n{0}\n'.format(list(data_set.keys())))
print('\nTipos dos dados:\n{0}\n'.format(data_set.dtypes))
print('Percentual médio de dados ausentes:', round(nan_analyze.mean()*100,1),'%')

# Gráfico com Percentual de dados ausentes 
label_perc = []

for i in np.arange(0, len(data_set.columns), 10):
    label_perc.append(str(i)+"%")
plt.figure(figsize=[10,40])

plt.yticks(np.arange(len(data_set.columns)), nan_analyze.index.values)
plt.xticks(np.arange(0, 1.1, .1), label_perc)

plt.ylim(0,len(data_set.columns))

plt.barh(np.arange(len(data_set.columns)), nan_analyze)



"""### ESTATÍSTICA DESCRITIVA DOS DADOS
"""

# Exibe apenas os campos numéricos:

print(data_set.describe())

# Para se ter uma visão dos atributos categóricos, os atributos não numéricos são descartados.

categ = data_set.dtypes[data_set.dtypes == 'object'].index

print("\n", data_set [categ].describe(), sep='\n')


# Limpeza da base de dados 
data_set_filtered = data_set[~np.isnan(data_set['Hematocrit'])]
nan_analyze_filtered = data_set_filtered.isna().sum()/len(data_set_filtered)

print('Quantidade de tuplas:', len(data_set_filtered))
print('Percentual médio de dados ausentes:', round(nan_analyze_filtered.mean()*100,1),"%")

# Gráfico com parâmetros que possuem menos de 40% de dados ausentes 
label_perc = []
for i in np.arange(0, 110, 10):
    label_perc.append(str(i)+"%")
plt.figure(figsize=[10,40])

plt.yticks(np.arange(len(data_set_filtered.columns)), nan_analyze_filtered.index.values)
plt.xticks(np.arange(0, 1.1, .1), label_perc)

plt.ylim(0,len(data_set_filtered.columns))

plt.barh(np.arange(len(data_set_filtered.columns)), nan_analyze_filtered)

data_set_filtered = data_set_filtered[nan_analyze_filtered[nan_analyze_filtered<=.4].index.values]

nan_analyze_filtered = data_set_filtered.isna().sum()/len(data_set_filtered)

print('Quantidade de tuplas:', len(data_set_filtered))
print('Percentual médio de dados ausentes:', round(nan_analyze_filtered.mean()*100,1),'%')

label_perc = []
for i in np.arange(0, 110, 10):
    label_perc.append(str(i)+'%')
plt.figure(figsize=[10,10])

plt.yticks(np.arange(len(data_set_filtered.columns)), nan_analyze_filtered.index.values)
plt.xticks(np.arange(0, 1.1, .1), label_perc)

plt.ylim(0,len(data_set_filtered.columns))

plt.barh(np.arange(len(data_set_filtered.columns)), nan_analyze_filtered)

# Escolhas dos parâmetros

data = data_set_filtered[['SARS-Cov-2 exam result', 'Hematocrit', 'Hemoglobin', 'Platelets', 'Mean platelet volume ', 'Red blood Cells', 'Lymphocytes', 'Mean corpuscular hemoglobin concentration (MCHC)', 'Leukocytes', 'Basophils', 'Mean corpuscular hemoglobin (MCH)', 'Eosinophils', 'Mean corpuscular volume (MCV)', 'Monocytes', 'Red blood cell distribution width (RDW)', 'Neutrophils']]

data.info()

print(data.describe())

# Transformando os dados categóricos do atributo SARS-Cov-2 exam result em dados binários
data_resultado = pd.get_dummies(data['SARS-Cov-2 exam result'])

data = pd.concat((data_resultado, data), axis=1) 
data = data.drop(['negative'], axis=1) 
data = data.drop(['SARS-Cov-2 exam result'], axis=1)
data = data.rename(columns={'positive': 'SARS-Cov-2 exam result'}) 

# Dados ausentes 
print(data.isnull().sum())

# Preenchimento dos valores ausentes com o valor da mediana 
features = data.dtypes[data.dtypes=='float64'].index.values
for feature in features:
  median = data[feature].median()
  data[feature] = data[feature].fillna(median)

# Boxplot para identificação de outliers 
df = data[data.dtypes[data.dtypes=='float64'].index.values]
df.boxplot(rot=90)

# Eritograma - Distribuição dos Dados
df = data[['SARS-Cov-2 exam result','Hematocrit', 'Hemoglobin', 'Mean corpuscular volume (MCV)']]
sns.set(style='ticks')
sns.pairplot(df, hue='SARS-Cov-2 exam result');

# Leucograma - Distribuição dos Dados 
df = data[['SARS-Cov-2 exam result','Leukocytes', 'Basophils', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Neutrophils']]
sns.set(style='ticks')
sns.pairplot(df, hue='SARS-Cov-2 exam result');

# Plaquetograma - Distribuição dos Dados
df = data[['SARS-Cov-2 exam result','Platelets', 'Mean platelet volume ']]
sns.set(style="ticks")
sns.pairplot(df, hue='SARS-Cov-2 exam result');

# Quantidade de Positivos e Negativos 
data.groupby('SARS-Cov-2 exam result').size()

# Gráfico Quantidade de Positivos e Negativos 
ax = data['SARS-Cov-2 exam result'].value_counts().plot(kind='bar', figsize=(14,8))
ax.set_xticklabels(['Negativo', 'Positivo'], rotation=0, fontsize=20)

# Diferença entre as Distribuições de casos Positivos e Negativos
features = data.dtypes[data.dtypes=='float64'].index.values

positive = data[data['SARS-Cov-2 exam result']==1]
negative = data[data['SARS-Cov-2 exam result']==0]
ks_list = []
pvalue_list = []
feature_list = []

for feature in features:    
    if len(positive)*len(negative)>0:
        ks, pvalue = ks_2samp(positive[feature], negative[feature])
        ks_list.append(ks)
        pvalue_list.append(pvalue)
        feature_list.append(feature)
        
df_ks = pd.DataFrame(data=zip(ks_list,pvalue_list),columns=['ks', 'pvalue'],index=feature_list)
df_ks = df_ks.sort_values(by='ks',ascending=True)

df_ks['ks']
plt.figure(figsize=(8,15))
plt.yticks(np.arange(len(df_ks)), df_ks.index.values)
plt.barh(np.arange(len(df_ks)), df_ks['ks'])

# Matriz de correlação dos parâmetros
sns.heatmap(data.corr(), annot=True, annot_kws = {'size':5}, cmap='Blues')

# CONJUNTO DE DADOS DESBALANCEADOS
# Separa os atributos e a classe do conjunto de dados em X e y
X = data.values[:,1:]
y = data.values[:,0]

# definindo uma seed global 
np.random.seed(1) 

# BALANCEAMENTO DOS DADOS
# definindo a estratégia de sub-amostragem
undersample = RandomUnderSampler(sampling_strategy=1)

# Aplicando a transformação de reamostragem
X_under, y_under = undersample.fit_resample(X,y)

# definindo a estratégia de super-amostragem
oversample = SMOTE(sampling_strategy=1)

# aplicando a transformação de reamostragem
X_over, y_over = oversample.fit_resample(X, y)

# Total de Resultados de cada conjunto
print('\ny:', Counter(y))
print('\ny_under:', Counter(y_under))
print('\ny_over:', Counter(y_over))

# Particionando o conjunto de dados desbalanceado em conjuntos de treino (80%) e teste (20%) mantendo a proporção de classes
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Particionando os conjuntos de dados balanceados pelas técnicas em conjuntos treino (80%) e teste (20%) mantendo a proporção de classes
# Random Undersampling
X_under_train, X_under_test, Y_under_train, Y_under_test = train_test_split(X_under, y_under, test_size=0.2, stratify=y_under)

# SMOTE Oversampling
X_over_train, X_over_test, Y_over_train, Y_over_test = train_test_split(X_over, y_over, test_size=0.2, stratify=y_over)

# Mostrando importância de cada feature nos conjuntos de dados de treinamento usando o algoritmo Floresta Aleatória

# Conjunto de Dados Desbalanceado
model = RandomForestClassifier()
model.fit(X_train, Y_train)
model.feature_importances_

importances = pd.Series(model.feature_importances_, index=data.columns[1:])
sns.barplot(x=importances, y=importances.index, orient='h', color='Blue')

# Conjunto de Dados Balanceado pela Técnica Random Undersampling
model_under = RandomForestClassifier()
model_under.fit(X_under_train, Y_under_train)
model_under.feature_importances_

importances = pd.Series(model_under.feature_importances_, index=data.columns[1:])
sns.barplot(x=importances, y=importances.index, orient='h', color='Blue')

# Conjunto de Dados Balanceado pela Técnica SMOTE Oversampling
model_over = RandomForestClassifier()
model_over.fit(X_over_train, Y_over_train)
model_over.feature_importances_

importances = pd.Series(model_over.feature_importances_, index=data.columns[1:])
sns.barplot(x=importances, y=importances.index, orient='h',  color='Blue')


# Criando modelo Árvore de Decisão para cada Conjunto de Dados
model_dtc = DecisionTreeClassifier()
model_dtc_under = DecisionTreeClassifier()
model_dtc_over = DecisionTreeClassifier()

# Treinando com Conjunto de Dados Desbalanceado
model_dtc.fit(X_train, Y_train)
# Treinando com Conjunto de Dados Balanceado pela Técnica Random Undersampling
model_dtc_under.fit(X_under_train, Y_under_train)
# Treinando com Conjunto de Dados Balanceado pela Técnica SMOTE Oversampling
model_dtc_over.fit(X_over_train, Y_over_train)

# Fazendo as predições no conjunto de teste
Y_predito = model_dtc.predict(X_test)
Y_predito_under = model_dtc_under.predict(X_test)
Y_predito_over = model_dtc_over.predict(X_test)

# métricas de avaliação das predições
print('Acurácia: %.2f' % accuracy_score(Y_test, Y_predito))
print('Precisão: %.2f' % precision_score(Y_test, Y_predito))
print('Recall: %.2f' % recall_score(Y_test, Y_predito))
print('F-score: %.2f' % f1_score(Y_test, Y_predito))
print('\n')
print('Acurácia: %.2f' % accuracy_score(Y_test, Y_predito_under))
print('Precisão: %.2f' % precision_score(Y_test, Y_predito_under))
print('Recall: %.2f' % recall_score(Y_test, Y_predito_under))
print('F-score: %.2f' % f1_score(Y_test, Y_predito_under))
print('\n')
print('Acurácia: %.2f' % accuracy_score(Y_test, Y_predito_over))
print('Precisão: %.2f' % precision_score(Y_test, Y_predito_over))
print('Recall: %.2f' % recall_score(Y_test, Y_predito_over))
print('F-score: %.2f' % f1_score(Y_test, Y_predito_over))

ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_predito)).plot()
ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_predito_under)).plot()
ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_predito_over)).plot()


# Criando modelo Regressão Logística para cada Conjunto de Dados
model_lr = LogisticRegression()
model_lr_under = LogisticRegression()
model_lr_over = LogisticRegression()

# Treinando com Conjunto de Dados Desbalanceado
model_lr.fit(X_train, Y_train)
# Treinando com Conjunto de Dados Balanceado pela Técnica Random Undersampling
model_lr_under.fit(X_under_train, Y_under_train)
# Treinando com Conjunto de Dados Balanceado pela Técnica SMOTE Oversampling
model_lr_over.fit(X_over_train, Y_over_train)

# Fazendo as predições no conjunto de teste
Y_predito = model_lr.predict(X_test)
Y_predito_under = model_lr_under.predict(X_test)
Y_predito_over = model_lr_over.predict(X_test)

# métricas de avaliação das predições
print('Acurácia: %.2f' % accuracy_score(Y_test, Y_predito))
print('Precisão: %.2f' % precision_score(Y_test, Y_predito))
print('Recall: %.2f' % recall_score(Y_test, Y_predito))
print('F-score: %.2f' % f1_score(Y_test, Y_predito))
print('\n')
print('Acurácia: %.2f' % accuracy_score(Y_test, Y_predito_under))
print('Precisão: %.2f' % precision_score(Y_test, Y_predito_under))
print('Recall: %.2f' % recall_score(Y_test, Y_predito_under))
print('F-score: %.2f' % f1_score(Y_test, Y_predito_under))
print('\n')
print('Acurácia: %.2f' % accuracy_score(Y_test, Y_predito_over))
print('Precisão: %.2f' % precision_score(Y_test, Y_predito_over))
print('Recall: %.2f' % recall_score(Y_test, Y_predito_over))
print('F-score: %.2f' % f1_score(Y_test, Y_predito_over))

ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_predito)).plot()
ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_predito_under)).plot()
ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_predito_over)).plot()


# Criando modelo Floresta Aleatória para cada Conjunto de Dados
model_rfc = RandomForestClassifier()
model_rfc_under = RandomForestClassifier()
model_rfc_over = RandomForestClassifier()

# Treinando com Conjunto de Dados Desbalanceado
model_rfc.fit(X_train, Y_train)
# Treinando com Conjunto de Dados Balanceado pela Técnica Random Undersampling
model_rfc_under.fit(X_under_train, Y_under_train)
# Treinando com Conjunto de Dados Balanceado pela Técnica SMOTE Oversampling
model_rfc_over.fit(X_over_train, Y_over_train)

# Fazendo as predições no conjunto de teste
Y_predito = model_rfc.predict(X_test)
Y_predito_under = model_rfc_under.predict(X_test)
Y_predito_over = model_rfc_over.predict(X_test)

# métricas de avaliação das predições
print('Acurácia: %.2f' % accuracy_score(Y_test, Y_predito))
print('Precisão: %.2f' % precision_score(Y_test, Y_predito))
print('Recall: %.2f' % recall_score(Y_test, Y_predito))
print('F-score: %.2f' % f1_score(Y_test, Y_predito))
print('\n')
print('Acurácia: %.2f' % accuracy_score(Y_test, Y_predito_under))
print('Precisão: %.2f' % precision_score(Y_test, Y_predito_under))
print('Recall: %.2f' % recall_score(Y_test, Y_predito_under))
print('F-score: %.2f' % f1_score(Y_test, Y_predito_under))
print('\n')
print('Acurácia: %.2f' % accuracy_score(Y_test, Y_predito_over))
print('Precisão: %.2f' % precision_score(Y_test, Y_predito_over))
print('Recall: %.2f' % recall_score(Y_test, Y_predito_over))
print('F-score: %.2f' % f1_score(Y_test, Y_predito_over))

ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_predito)).plot()
ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_predito_under)).plot()
ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_predito_over)).plot()



# Criando modelo SVM para cada Conjunto de Dados
model_svm = svm.SVC()
model_svm_under = svm.SVC()
model_svm_over = svm.SVC()

# Treinando com Conjunto de Dados Desbalanceado
model_svm.fit(X_train, Y_train)
# Treinando com Conjunto de Dados Balanceado pela Técnica Random Undersampling
model_svm_under.fit(X_under_train, Y_under_train)
# Treinando com Conjunto de Dados Balanceado pela Técnica SMOTE Oversampling
model_svm_over.fit(X_over_train, Y_over_train)

# Fazendo as predições no conjunto de teste
Y_predito = model_svm.predict(X_test)
Y_predito_under = model_svm_under.predict(X_test)
Y_predito_over = model_svm_over.predict(X_test)

# métricas de avaliação das predições
print('Acurácia: %.2f' % accuracy_score(Y_test, Y_predito))
print('Precisão: %.2f' % precision_score(Y_test, Y_predito))
print('Recall: %.2f' % recall_score(Y_test, Y_predito))
print('F-score: %.2f' % f1_score(Y_test, Y_predito))
print('\n')
print('Acurácia: %.2f' % accuracy_score(Y_test, Y_predito_under))
print('Precisão: %.2f' % precision_score(Y_test, Y_predito_under))
print('Recall: %.2f' % recall_score(Y_test, Y_predito_under))
print('F-score: %.2f' % f1_score(Y_test, Y_predito_under))
print('\n')
print('Acurácia: %.2f' % accuracy_score(Y_test, Y_predito_over))
print('Precisão: %.2f' % precision_score(Y_test, Y_predito_over))
print('Recall: %.2f' % recall_score(Y_test, Y_predito_over))
print('F-score: %.2f' % f1_score(Y_test, Y_predito_over))

ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_predito)).plot()
ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_predito_under)).plot()
ConfusionMatrixDisplay(confusion_matrix(Y_test, Y_predito_over)).plot()
