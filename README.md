# 🏠 Previsão de Preços de Imóveis — EUA

Modelo de **regressão linear múltipla** construído do zero, sem bibliotecas de machine learning, utilizando gradiente descendente para prever preços de imóveis com base no [USA Housing Dataset](https://www.kaggle.com/datasets/fratzcan/usa-house-prices) do Kaggle.

---

## 📌 Visão Geral

O projeto implementa todo o pipeline de um modelo de regressão linear manualmente: desde o carregamento e limpeza dos dados até o treinamento por gradiente descendente e a visualização dos resultados. Somente **NumPy**, **Pandas** e **Matplotlib** são utilizados.

---

## 🧮 Fundamentos Matemáticos

### 1. O Modelo Linear

A regressão linear múltipla assume que o preço de um imóvel pode ser aproximado por uma combinação linear de suas características:

$$\hat{y} = \mathbf{X} \cdot \mathbf{m} + b$$

Onde:
- $\hat{y}$ — vetor de preços previstos
- $\mathbf{X}$ — matriz de features $(n\_samples \times n\_features)$
- $\mathbf{m}$ — vetor de pesos/coeficientes $(n\_features)$
- $b$ — bias (intercepto)

O objetivo é encontrar os valores de $\mathbf{m}$ e $b$ que fazem $\hat{y}$ se aproximar ao máximo dos preços reais $y$.

---

### 2. Função de Erro — MSE

Para medir o quão errado o modelo está, usamos o **Erro Quadrático Médio** (Mean Squared Error):

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- Eleva os erros ao quadrado para penalizar previsões muito distantes
- Sempre positivo — quanto menor, melhor
- Diferenciável, o que permite usar gradiente descendente para minimizá-lo

---

### 3. Normalização (Z-score)

Antes do treinamento, cada feature é normalizada:

$$X_{norm} = \frac{X - \mu}{\sigma}$$

Onde $\mu$ é a média e $\sigma$ é o desvio padrão de cada coluna.

**Por que normalizar?**
Features em escalas muito diferentes (ex: `sqft_living` em milhares vs `bedrooms` em unidades) fazem o gradiente descendente convergir lentamente ou de forma instável. Após a normalização, todas as features ficam na mesma escala (média 0, desvio 1), e o algoritmo converge muito mais rápido.

> ⚠️ Os valores de $\mu$ e $\sigma$ são calculados **apenas no treino** e reutilizados na predição do teste, evitando vazamento de dados.

---

### 4. Gradiente Descendente

O gradiente descendente é um algoritmo iterativo de otimização. A ideia é: dado que o MSE é uma função de $\mathbf{m}$ e $b$, podemos caminhar na direção oposta ao gradiente (a "descida mais íngreme") para minimizá-lo.

A cada iteração, os parâmetros são atualizados assim:

$$\mathbf{m} \leftarrow \mathbf{m} - \alpha \cdot \frac{\partial MSE}{\partial \mathbf{m}}$$

$$b \leftarrow b - \alpha \cdot \frac{\partial MSE}{\partial b}$$

Onde $\alpha$ é a **taxa de aprendizado** (learning rate), que controla o tamanho do passo.

#### Derivadas do MSE

Derivando $MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$ em relação a cada parâmetro:

$$\frac{\partial MSE}{\partial \mathbf{m}} = -\frac{2}{n} \mathbf{X}^T \cdot (y - \hat{y})$$

$$\frac{\partial MSE}{\partial b} = -\frac{2}{n} \sum (y_i - \hat{y}_i)$$

No código isso se traduz diretamente em:

```python
dm = -(2/n_sample) * np.dot(X.T, (y - y_pred))
db = -(2/n_sample) * np.sum(y - y_pred)

m -= learning_rate * dm
b -= learning_rate * db
```

Após `iterations` repetições, $\mathbf{m}$ e $b$ convergem para os valores que minimizam o MSE.

---

### 5. Predição

Para prever um novo dado, basta aplicar a mesma normalização do treino e calcular o produto interno:

$$\hat{y}_{novo} = \mathbf{X}_{norm} \cdot \mathbf{m} + b$$

---

## ⚙️ Features Utilizadas

Após análise de correlação com `price` via `df.corr()`, as seguintes colunas foram **mantidas**:

| Feature          | Descrição                        |
|------------------|----------------------------------|
| `bedrooms`       | Número de quartos                |
| `bathrooms`      | Número de banheiros              |
| `sqft_living`    | Área interna em pés quadrados    |
| `floors`         | Número de andares                |
| `waterfront`     | Vista para orla (0 ou 1)         |
| `view`           | Pontuação de vista               |
| `sqft_above`     | Área acima do solo               |
| `sqft_basement`  | Área do porão                    |

Colunas **removidas** por baixa correlação ou por serem não-numéricas:
`date`, `street`, `city`, `statezip`, `country`, `sqft_lot`, `condition`, `yr_renovated`, `yr_built`

---

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install pandas numpy matplotlib kagglehub
```

### Dataset

Baixe o dataset pelo KaggleHub ou manualmente e atualize a variável `path` no script com o caminho correto do arquivo `USA Housing Dataset.csv`.

### Execução

```bash
python model.py
```

---

## 📊 Resultado

Um gráfico de dispersão é exibido comparando os preços **reais** (eixo X) com os **previstos** (eixo Y). A linha vermelha representa a predição perfeita — quanto mais os pontos se concentrarem sobre ela, melhor o modelo.

---

## 🔧 Hiperparâmetros

| Parâmetro        | Valor  | Descrição                                      |
|------------------|--------|------------------------------------------------|
| `learning_rate`  | 0.001  | Tamanho do passo em cada iteração              |
| `iterations`     | 10.000 | Número de iterações do gradiente descendente   |
| Função de perda  | MSE    | Erro Quadrático Médio                          |
| Normalização     | Z-score| Média 0 e desvio padrão 1 por feature          |

---

## 📖 Conceitos Abordados

- Regressão linear múltipla
- Gradiente descendente (batch)
- Derivadas parciais e cálculo do gradiente
- Normalização de features (Z-score)
- Seleção de features por correlação
- Divisão treino/teste
- Avaliação visual do modelo
