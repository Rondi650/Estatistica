# 📘 Guia de Referência Rápida: ISLP & People Analytics

Documento de apoio para o ecossistema de estudos no VS Code. Reúne notações estatísticas, equivalências em Python/Excel, conceitos de validação de modelos e o cronograma de estudos focado em People Analytics.

---

## 1. Notação Matemática e Tradução para Código

### 1.1 Equação Fundamental
$$Y = f(X) + \epsilon$$

* **$Y$**: Variável resposta / Alvo (ex: Salário, Desligamento).
* **$X$**: Preditores / Recursos / Variáveis de entrada (ex: Tempo de Casa, Avaliação de Desempenho).
* **$f$**: A função/regra real e oculta que conecta $X$ a $Y$. No Python, representa a lógica que queremos aproximar: `def f(X)`.
* **$\hat{f}$ ($f$-chapéu)**: O modelo estatístico ou algoritmo treinado (`model.fit(X, y)`).
* **$\hat{y}$ ($y$-chapéu)**: A previsão gerada pelo modelo ($\hat{y} = \hat{f}(X)$).
* **$\epsilon$ (Épsilon)**: Ruído imprevisível (erro irredutível inerente ao processo real).

---

### 1.2 Dimensões e Índices

| Símbolo | Significado em Estatística | Equivalente no Excel / Python |
| :--- | :--- | :--- |
| **$n$** | Número total de observações (linhas) | `len(df)` ou `df.shape[0]` |
| **$p$** | Número total de preditores (colunas) | `df.shape[1] - 1` |
| **$i$** | Índice da linha ($i = 1, \dots, n$) | Índice de linha: `df.iloc[i]` |
| **$j$** | Índice da coluna ($j = 1, \dots, p$) | Índice de coluna: `df.iloc[:, j]` |
| **$\mathbb{R}^{n \times p}$** | Matriz de números reais com $n$ linhas e $p$ colunas | `X.shape == (n, p)` |

---

### 1.3 Convenções Tipográficas no ISLP

* **MAIÚSCULA NEGRITO ($\mathbf{X}$)**: Matriz (tabela 2D de tamanho $n \times p$).
* **minúscula negrito ($\mathbf{y}$ ou $\mathbf{x}_i$)**: Vetor (coluna com $n$ elementos ou linha com $p$ elementos).
* **minúscula normal ($a, x_{ij}$)**: Escalar (número único de uma célula específica).
* **MAIÚSCULA NORMAL ($X, Y$)**: Variável aleatória (conceito teórico da variável).

---

### 1.4 Notação por Observação (Treino vs. Teste)

```text
       LINHA DE TREINO (Dados Históricos)       │        LINHA DE TESTE / NOVO DADO (Produção)
                                                 │
    x_i  = Ficha inteira da pessoa i (linha)     │     x_0  = Ficha inteira de uma pessoa nova (X_test)
    y_i  = Gabarito real da pessoa i             │     y_0  = Gabarito real da pessoa nova (y_test)
 f_hat(x_i) = Previsão do modelo para a pessoa i │  f_hat(x_0) = Previsão do modelo para a pessoa nova
```

* **$\hat{f}(x_i) \approx y_i$**: Objetivo do treino — fazer a previsão para cada colaborador histórico chegar o mais perto possível do valor real dele.
* **$\hat{f}(x_0) \approx y_0$**: Objetivo de generalização — fazer o modelo prever com precisão para uma observação nova ($x_0$) que **não estava** na base de treino.

---

## 2. Como o Aprendizado Estatístico Funciona

O aprendizado estatístico busca estimar uma relação desconhecida entre preditores e uma resposta.

### 2.1 Duas perguntas principais

**Predição:**

Queremos uma previsão boa para uma pessoa ou situação nova. Neste caso, $\hat{f}$ é importante, mas não precisamos necessariamente entender toda a relação entre as variáveis.

Exemplo: prever se um colaborador tem alto risco de desligamento.

**Inferência:**

Queremos entender como $X$ está associado com $Y$. Aqui a forma de $\hat{f}$ é importante.

Exemplo: investigar quais fatores estão associados ao salário e em qual direção.

Muitas aplicações podem exigir os dois objetivos.

### 2.2 Métodos paramétricos

Um método paramétrico começa por uma forma assumida para $f$ e depois estima seus parâmetros.

Exemplo de regressão linear:

$$
\hat{f}(X)=\hat{\beta}_0+\hat{\beta}_1X_1+\dots+\hat{\beta}_pX_p
$$

Os coeficientes $\hat{\beta}_0,\dots,\hat{\beta}_p$ são os parâmetros que o modelo precisa estimar.

Etapas:

1. Escolher uma forma ou formato para $f$.
2. Usar os dados de treino para estimar os parâmetros.
3. Usar $\hat{f}$ para fazer previsões.

Vantagens: geralmente são mais simples e interpretáveis.

Desvantagens: se a forma assumida estiver muito longe da realidade, o modelo pode ser inadequado.

### 2.3 Métodos não paramétricos

Métodos não paramétricos não assumem uma forma específica para $f$. Eles tentam aproximar a função diretamente a partir dos dados.

Exemplo: $K$-vizinhos mais próximos (KNN).

Para prever uma observação nova, o KNN procura as $K$ observações de treino mais próximas dela e calcula uma estimativa a partir desses vizinhos.

- KNN com $K$ pequeno tende a produzir uma função mais flexível.
- KNN com $K$ grande produz uma função mais suave.
- Quanto maior $K$, normalmente menor a flexibilidade.

Vantagens: podem representar relações complexas sem assumir uma fórmula prévia.

Desvantagens: exigem mais dados, são mais difíceis de interpretar e podem piorar em bases com muitos preditores.

### 2.4 Aprendizado supervisionado e não supervisionado

**Supervisionado:**

Os dados possuem uma resposta $Y$ conhecida.

Exemplos:

- prever salário;
- prever desligamento;
- prever nota de desempenho.

**Não supervisionado:**

Não existe uma resposta conhecida para orientar o treino.

Exemplo: agrupar colaboradores com perfis semelhantes de carreira ou comportamento.

### 2.5 Regressão e classificação

A principal diferença não está no algoritmo isoladamente, mas no tipo da variável resposta.

**Regressão:** $Y$ é quantitativa ou contínua.

Exemplos: salário, tempo de casa, horas de treinamento.

**Classificação:** $Y$ pertence a categorias.

Exemplos: permanece/desliga, aprovado/reprovado, risco alto/baixo.

---

## 3. Erro Redutível, Irredutível e Qualidade da Estimativa

A relação real pode ser escrita como:

$$
Y=f(X)+\epsilon
$$

O erro esperado de uma previsão pode ser decomposto em:

$$
\text{Erro esperado}
=\text{Erro redutível}
+\text{Erro irredutível}
$$

**Erro redutível:** é produzido pela aproximação inadequada de $f$ por $\hat{f}$.

O treino busca reduzir esse erro.

**Erro irredutível:** está associado ao termo $\epsilon$. Ele representa informação ausente, aleatoriedade ou variáveis não observadas.

Não podemos eliminá-lo escolhendo apenas um algoritmo melhor.

Exemplo: duas pessoas com o mesmo tempo de casa e salário podem ter resultados diferentes de desligamento por fatores que não estão na base.

---

## 4. Métricas de Desempenho e Avaliação de Erro

As métricas abaixo são usadas principalmente para respostas quantitativas.

### 4.1 Principais Métricas

* **Resíduo ($e_i$)**: $e_i = y_i - \hat{y}_i$ (diferença entre o real e o previsto).
* **RSS (Residual Sum of Squares)**:
  $$\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{f}(x_i))^2$$
* **MSE (Mean Squared Error)**:
  $$\text{MSE} = \frac{1}{n} \text{RSS} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{f}(x_i))^2$$
* **RMSE (Root Mean Squared Error)**: $\text{RMSE} = \sqrt{\text{MSE}}$ (retorna o erro para a unidade original, ex: R$).
* **$R^2$**: Proporção da variância explicada pelo modelo ($0 \le R^2 \le 1$).

---

### 4.2 Test MSE vs. Training MSE

**O que são dados de treino e dados de teste?**

Antes de treinar um modelo, separamos os dados disponíveis em dois grupos:

- **Dados de treino:** exemplos que o modelo usa para aprender padrões e ajustar seus parâmetros. É como estudar com exercícios e respostas.
- **Dados de teste:** exemplos separados, que o modelo não usa durante o treino. Servem para avaliar como ele se sai diante de casos novos, como uma prova.

Por exemplo, podemos treinar com dados históricos de alguns colaboradores e testar o modelo com outros colaboradores que ficaram de fora do treino. Para que a avaliação seja justa, o modelo não deve aprender com as respostas dos dados de teste antes de ser avaliado.

* **Training MSE**:
  $$\text{Training MSE} = \text{Ave}_{i \in \text{treino}} (y_i - \hat{f}(x_i))^2$$
* **Test MSE**:
  $$\text{Test MSE} = \text{Ave}_{0 \in \text{teste}} (y_0 - \hat{f}(x_0))^2$$

> **Por que o Training MSE é quase sempre menor que o Test MSE?**
> Porque o algoritmo ajusta os parâmetros (pesos $\beta$) otimizando diretamente a redução dos erros da base de treino. Ao avaliar em dados inéditos ($x_0$), o erro natural tende a ser um pouco maior.

```text
                    COMPORTAMENTO DOS ERROS SEGUNDO O MODELO
                    
      Erro (MSE) 
          │
          │         \             / (Test MSE dispara = Overfitting)
          │          \___________/   <-- Test MSE (Ave(y_0 - f_hat(x_0))²)
          │           \_________
          │                     \___ <-- Training MSE (Ave(y_i - f_hat(x_i))²)
          └─────────────────────────────► Flexibilidade do Modelo
```

---

## 5. Flexibilidade, Viés e Variância

**Flexibilidade** indica a capacidade de um método acompanhar padrões complexos nos dados.

- Pouca flexibilidade: modelo mais simples e suave.
- Muita flexibilidade: modelo pode se ajustar a padrões locais dos dados.

**Viés:**

O viés mede o quanto o método erra por simplificar a realidade.

Exemplo: ajustar uma relação curva por uma linha reta pode produzir um viés alto.

**Variância:**

A variância mede o quanto $\hat{f}$ mudaria se o modelo fosse treinado com outra amostra de dados.

Um método muito flexível pode mudar muito entre amostras e apresentar variância alta.

**Trade-off viés e variância:**

- Pouca flexibilidade: tende a aumentar o viés e diminuir a variância.
- Muita flexibilidade: tende a diminuir o viés e aumentar a variância.
- O objetivo não é eliminar ambos, mas encontrar um nível de flexibilidade que minimize o Test MSE.

**Underfitting:**

O modelo é simples demais e não representa bem nem os padrões do treino.

**Overfitting:**

O modelo se ajusta demais aos dados de treino e aprende ruído específico daquela amostra. O Training MSE pode ficar baixo, enquanto o Test MSE fica alto.

---

## 6. Avaliação de Classificação

Para uma resposta categórica, o MSE não é a métrica principal.

**Taxa de erro de treino:**

$$
\text{Erro}_{\text{treino}}
=\frac{1}{n}\sum_{i=1}^{n}I(y_i\neq\hat{y}_i)
$$

$I(\cdot)$ vale $1$ quando a condição é verdadeira e $0$ caso contrário.

Exemplo: em $100$ colaboradores, $12$ previsões são incorretas. A taxa de erro é $12/100=0{,}12$ ou $12\%$.

**Taxa de acerto:**

$$
\text{Acurácia}=1-\text{Taxa de erro}
$$

Para problemas desbalanceados, a acurácia pode enganar. Por exemplo, prever que ninguém desligará em uma base com apenas $5\%$ de desligamentos pode resultar em $95\%$ de acurácia, mas o modelo não identifica nenhum desligamento.

### 6.1 Classificador de Bayes

O classificador de Bayes atribui uma observação à classe com maior probabilidade condicional.

Exemplo:

$$
P(Y=\text{desligamento}\mid X=x)>P(Y=\text{permanece}\mid X=x)
$$

Nesse caso, o classificador de Bayes prevê desligamento.

Esse classificador é teoricamente ótimo, mas suas probabilidades reais geralmente são desconhecidas.

### 6.2 $K$-vizinhos mais próximos para classificação

O KNN estima a probabilidade de cada classe usando os vizinhos mais próximos.

Exemplo com $K=5$, em que 3 vizinhos desligaram e 2 permaneceram:

- probabilidade estimada de desligamento: $3/5=0{,}60$;
- probabilidade estimada de permanência: $2/5=0{,}40$;
- previsão: desligamento.

Pequenos valores de $K$ produzem fronteiras de classificação mais irregulares e maior variância. Grandes valores produzem fronteiras mais suaves e maior viés.

---

## 7. Estrutura e Roteiro de Leitura do ISLP

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MAPEAMENTO DOS CAPÍTULOS ISLP                         │
├───────────────────────────────────┬─────────────────────────────────────────┤
│ Capítulos 2, 3, 4 e 5             │ OBRIGATÓRIO (Básico e Fundamentos)      │
│   • Cap 2: Statistical Learning   │   - Regressão Linear Múltipla           │
│   • Cap 3: Linear Regression      │   - Regressão Logística (Classificação) │
│   • Cap 4: Classification         │   - Validação Cruzada / Resampling      │
│   • Cap 5: Resampling Methods     │                                         │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ Capítulos 6 e 7                   │ PULAR AGORA (Avançado/Não-linear)       │
│   • Cap 6: Linear Model Selection │   - Exige cálculo pesados (Ridge/Lasso) │
│   • Cap 7: Moving Beyond Linearity│   - Retomar no 2º Semestre/2027         │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ Capítulo 8                        │ LEITURA DIRETA (Pular do 5 para o 8)    │
│   • Cap 8: Tree-Based Methods     │   - Árvores de Decisão, Random Forest   │
│                                   │   - Alta interpretabilidade para RH     │
└───────────────────────────────────┴─────────────────────────────────────────┘
```

---

## 8. Integração: ISLP $\leftrightarrow$ People Analytics

| Capítulo ISLP | Livro de Apoio (*Fundamentals of PA*) | Aplicação Prática no RH |
| :--- | :--- | :--- |
| **Cap. 3 (Regressão)** | Capítulo de Análise de Remuneração | Estudo de Equidade Salarial e Drivers de Salário. |
| **Cap. 4 (Classificação)** | Capítulo de Turnover / Atrito | Modelo Logístico para prever Risco de Desligamento. |
| **Cap. 5 (Resampling)** | Avaliação de Modelos | Validação da estabilidade do modelo entre áreas/filiais. |
| **Cap. 8 (Árvores)** | Modelos Avançados de Pessoas | Regras claras de retenção para apresentação executiva. |

---

## 9. Cronograma e Organização do Estudo (Até Maio de 2027)

### 9.1 Estrutura Sugerida de Pastas no VS Code

```text
people-analytics-study/
├── docs/
│   └── islp_people_analytics_cheatsheet.md  <-- Este arquivo
├── datasets/
│   ├── raw/                                 <-- Bases brutas de RH (anonimizadas)
│   └── processed/                           <-- Bases limpas prontas para modelo
├── notebooks/
│   ├── 01_eda_salary_equity.ipynb           <-- Análise exploratória
│   ├── 02_linear_regression_salary.ipynb    <-- Cap. 3 ISLP + Fundamentals
│   ├── 03_logistic_turnover_risk.ipynb      <-- Cap. 4 ISLP + Fundamentals
│   └── 04_decision_trees_retention.ipynb    <-- Cap. 8 ISLP
└── src/
    └── utils.py                              <-- Funções auxiliares Python
```

---

### 5.2 Divisão da Rotina Diária (1h30 a 2h por dia)

* **Bloco 1: Leitura Ativa (45 a 50 min)**: Estudo teórico no livro (*ISLP* ou *Excellence/Fundamentals*).
* **Pausa (10 min)**: Descanso cognitivo.
* **Bloco 2: Mão na Massa (50 a 60 min)**: Reprodução dos laboratórios em Python dentro dos Jupyter Notebooks.

```text
PLANO REALISTA ATÉ MAIO DE 2027
- Concluir "Excellence in People Analytics" (Estratégia & ROI)
- Concluir "Storytelling com Dados" (Comunicação Executiva)
- Concluir ISLP (Capítulos 2, 3, 4, 5 e 8) em par com "Fundamentals of People Analytics"
- Entregar 1 a 2 projetos diagnósticos/preditivos aplicados no trabalho
```