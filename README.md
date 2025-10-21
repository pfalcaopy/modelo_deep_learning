# 🧠 Classificação Inteligente de Textos Financeiros com Deep Learning e GitHub Actions

Imagine ter milhares de registros financeiros — cada linha, um fragmento de texto com descrições, fornecedores e categorias.  
Agora imagine transformar essa massa de informações em **conhecimento estruturado**, de forma **automática**, **inteligente** e **constante**.  

Foi com esse propósito que nasceu este projeto.

---

## 🌱 A Ideia

O desafio começou simples: **categorizar automaticamente textos financeiros** de um extrato.  
Mas a ambição era maior — **criar um sistema autônomo**, capaz de:
- Aprender com novos dados do banco de dados;
- Se treinar novamente de forma automática;
- E manter o modelo sempre atualizado, sem intervenção manual.

Assim, unimos **Deep Learning** com **automação em nuvem via GitHub Actions**.

---

## ⚙️ O Coração do Projeto — O Modelo de Deep Learning

O arquivo [`model.py`](model.py) é o cérebro do sistema.

Ele inicia com uma conexão direta ao banco PostgreSQL, carregando os dados da tabela.  
A partir daí, o texto é transformado em algo que uma rede neural possa entender.

### 🧩 Pré-processamento e Criação do Texto Unificado
Cada linha do extrato é reconstruída como uma narrativa única:
```text
tipo_operacao + categoria + fornecedor + descricao + instituicao + categoria_financeira
```
Essas informações são convertidas para minúsculas e limpas, formando uma base textual rica em contexto.

### 🔢 Transformando Texto em Vetores
Utilizamos o poder do **TextVectorization** do TensorFlow:
```python
vectorizer = TextVectorization(max_tokens=5000, output_mode='int', output_sequence_length=100)
```
Isso converte palavras em números — a linguagem que as redes neurais compreendem.

---

## 🧬 A Arquitetura Neural

Para capturar o "ritmo" e a sequência das palavras, foi usada uma **rede recorrente LSTM (Long Short-Term Memory)**, capaz de compreender contexto e dependências entre termos.

```python
model = keras.Sequential([
    Embedding(input_dim=5000, output_dim=16, mask_zero=True),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

Essa arquitetura foi escolhida por equilibrar **profundidade e eficiência**, garantindo uma aprendizagem rápida e precisa mesmo com dados textuais complexos.

---

## 📈 O Aprendizado

Durante o treinamento, o modelo aprendeu a reconhecer padrões com velocidade impressionante.  
Nas primeiras épocas, a acurácia já ultrapassava **90%**, e logo alcançou **quase 100%**, tanto no treino quanto na validação.  
A perda (loss) caiu drasticamente nas primeiras iterações e estabilizou próxima de zero — sinal de **convergência ideal** e **excelente generalização**.

Esses resultados mostraram que a arquitetura, os hiperparâmetros e o pré-processamento estavam em harmonia.

---

## 💾 A Persistência do Conhecimento

Após o treinamento:
- O modelo é salvo como `modelo_classificacao.h5`;
- O codificador de rótulos (`LabelEncoder`) é salvo como `label_encoder.pkl`.

Esses arquivos tornam possível aplicar o modelo em produção, classificando novos textos em tempo real.

---

## 🤖 A Automação com GitHub Actions

O toque de mestre do projeto está no arquivo [`run-model.yaml`](run-model.yaml).  
Ele transforma o modelo em uma **entidade viva**, que se autoatualiza a cada 15 dias.

### 🔄 Workflow Automatizado
```yaml
on:
  schedule:
    - cron: "0 0 */15 * *"  # Executa a cada 15 dias
  workflow_dispatch: {}
```

O GitHub Actions:
1. Baixa o repositório;
2. Instala as dependências do [`requirements.txt`](requirements.txt);
3. Executa `python model.py` automaticamente.

Assim, a cada ciclo, o modelo é re-treinado com os **dados mais recentes** do banco, garantindo decisões cada vez mais alinhadas com a realidade financeira.

---

## 🧩 Tecnologias que Deram Vida ao Projeto

| Camada | Ferramenta |
|--------|-------------|
| Deep Learning | TensorFlow / Keras |
| Dados | Pandas, NumPy, SQLAlchemy |
| Banco de Dados | PostgreSQL |
| Automação | GitHub Actions |
| Linguagem | Python 3.11 |
| Utilidades | dotenv, scikit-learn, matplotlib |

---

## 🧪 Resultados e Impacto

O modelo mostrou **excelente desempenho**:
- Acurácia de validação próxima de 100%;
- Perda mínima após poucas épocas;
- Estabilidade em diferentes execuções.

Esses resultados indicam que o sistema **compreende profundamente a linguagem financeira** e está pronto para operação contínua.

---

## 👨‍💻 Autor

**🧔🏻 Pedro Falcão**  
🧩 Engenheiro de Dados  
💡  ETL, Eng. de dados, Analise de dados inteligente e sistemas autônomos  
📧 pfalcao.py@gmail.com

---

## 🕒 Em resumo

> Este projeto não é apenas um modelo de classificação —  
> é um **organismo digital** que aprende, evolui e se mantém atualizado sozinho.  
> Um exemplo real de como a inteligência artificial pode se integrar perfeitamente ao ciclo de automação moderna.

---

📜 **Os dados contêm uma história; nossa missão é narrá-las.** Pedro Falcão
