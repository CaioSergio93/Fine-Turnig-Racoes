import streamlit as st
import pandas as pd
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
# Importa as funções de utilidade
from utils import create_label_maps, tokenize_function, convert_to_datasets

# Configuração da página
st.set_page_config(page_title="Fine-Tuning Rações (Cães/Gatos) com Hugging Face", layout="centered")

st.title("🐾 Classificador de Perguntas sobre Rações")
st.write("Esta aplicação realiza o fine-tuning de um modelo BERT para classificar perguntas sobre rações para cães e gatos.")
st.markdown("---")

# Definições do Modelo
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
JSONL_FILE = "racoes_caes_gatos.jsonl"

# --- 1. Carregamento e Preparação dos Dados ---

@st.cache_data
def load_and_prepare_data(filepath):
    """
    Carrega dados do arquivo JSONL e cria uma classe simplificada (rótulo) 
    baseada em palavras-chave presentes na instrução.
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                instruction = item['instruction'].lower()
                
                # Lógica para extrair classe
                animal = 'outros'
                if 'cães' in instruction or 'cachorro' in instruction or 'cão' in instruction:
                    animal = 'cao'
                elif 'gatos' in instruction or 'gato' in instruction:
                    animal = 'gato'
                
                idade = 'geral'
                if 'filhotes' in instruction or 'filhote' in instruction:
                    idade = 'filhote'
                elif 'idosos' in instruction or 'idoso' in instruction or 'velhinho' in instruction:
                    idade = 'idoso'
                elif 'adultos' in instruction or 'adulto' in instruction:
                    idade = 'adulto'
                
                # A classe deve ser a combinação 'animal-idade'
                classe = f"{animal}-{idade}"
                
                # Filtra amostras que não pudemos rotular claramente (ex: 'outros-geral')
                if classe == "outros-geral":
                     continue
                
                # Adiciona o texto original da resposta para consulta posterior
                data.append({
                    'texto': item['instruction'], 
                    'classe': classe, 
                    'resposta_original': item['response'] 
                })

        df_loaded = pd.DataFrame(data)
        
        # Filtra classes com poucas amostras para evitar erro no train_test_split (stratify)
        class_counts = df_loaded['classe'].value_counts()
        # Aumentamos o requisito para 3 amostras por classe, se possível
        valid_classes = class_counts[class_counts >= 3].index 
        
        # Se o número de classes válidas for muito pequeno, diminuímos o requisito,
        # mas sempre garantimos que a classe de teste tenha pelo menos 1 amostra.
        if len(valid_classes) < 2:
            valid_classes = class_counts[class_counts >= 2].index
            
        df_filtered = df_loaded[df_loaded['classe'].isin(valid_classes)].reset_index(drop=True)
        
        return df_filtered

    except FileNotFoundError:
        st.error(f"Arquivo de dados não encontrado: {filepath}. O treinamento não será possível.")
        return pd.DataFrame() 
    except Exception as e:
        st.error(f"Erro ao carregar ou processar o JSONL: {e}")
        return pd.DataFrame()


df = load_and_prepare_data(JSONL_FILE)

if df.empty:
    st.error("❌ Não foi possível carregar dados válidos para treino. Por favor, certifique-se de que o arquivo `racoes_caes_gatos.jsonl` está no mesmo diretório.")
    st.stop()
else:
    st.info(f"✅ Dados carregados de `{JSONL_FILE}`. Total de {len(df)} amostras válidas após filtragem.")


st.subheader("Visualização dos Dados de Treino (Amostra)")
st.dataframe(df.head())
st.write(f"Contagem de Classes: {df['classe'].value_counts().to_dict()}") # Novo log para ver as classes

# Mapeamento de classes usando a função utilitária
df, label2id, id2label, labels = create_label_maps(df, class_column='classe')

# --- 2. Preparação para Fine-Tuning ---

# Separação de dados com stratify, pois agora temos mais dados
try:
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['texto'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
except ValueError as e:
    # Se houver erro de stratify, fazemos sem ele, mas avisamos
    st.warning(f"Aviso: Erro ao aplicar `stratify` na divisão dos dados ({e}). Tentando divisão simples.")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['texto'], df['label'], test_size=0.2, random_state=42)

# Conversão para Dataset usando a função utilitária
train_dataset, val_dataset = convert_to_datasets(train_texts, val_texts, train_labels, val_labels)

# Carregar Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    st.error(f"Erro ao carregar o tokenizer: {e}")
    st.stop()

# Tokenização usando a função utilitária
tokenize = tokenize_function(tokenizer)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Instanciar Modelo
if 'model_instance' not in st.session_state:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id
        )
        st.session_state.model_instance = model
    except Exception as e:
        st.error(f"Erro ao instanciar o modelo base: {e}")
        st.stop()

# --- 3. Treinamento ---

st.subheader("⚙️ Configurações e Treinamento")
st.write(f"Modelo base: `{MODEL_NAME}`")
st.write(f"Classes identificadas: `{', '.join(labels)}`")
st.write(f"Total de Amostras de Treino: `{len(train_dataset)}`")

if st.button("🚀 Iniciar Fine-Tuning"):
    with st.spinner("Treinando modelo... (O treinamento agora usará um dataset maior e será mais demorado.)"):
        training_args = TrainingArguments(
            output_dir="./results",
            logging_strategy="epoch", 
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=5,  # AUMENTADO PARA 5 EPOCHS
            weight_decay=0.01,
            logging_steps=10,
            save_total_limit=1,
            disable_tqdm=True,
            report_to="none"
        )

        trainer = Trainer(
            model=st.session_state.model_instance,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )

        trainer.train()

        # Salvar o estado do modelo treinado na sessão
        st.session_state.modelo_treinado = True
        st.session_state.trained_tokenizer = tokenizer
        st.session_state.trained_model_instance = st.session_state.model_instance


    st.success("✅ Modelo treinado com sucesso! Você já pode realizar previsões abaixo.")

# --- 4. Inferência ---

st.subheader("🔍 Faça uma previsão com o modelo treinado")
exemplo = st.text_area("Digite uma pergunta sobre rações (ex: Qual ração para gatos idosos com diabetes?)")

if st.button("Classificar", key="classificar_btn"):
    if not exemplo.strip():
        st.warning("Digite um texto para classificar.")
    elif 'modelo_treinado' not in st.session_state:
        st.error("O modelo ainda não foi treinado. Por favor, clique em '🚀 Iniciar Fine-Tuning' primeiro.")
    else:
        try:
            pipe = pipeline(
                "text-classification",
                model=st.session_state.trained_model_instance,
                tokenizer=st.session_state.trained_tokenizer,
                device=-1 
            )

            st.session_state.trained_model_instance.eval()

            with st.spinner("Classificando..."):
                resultado = pipe(exemplo)[0]
                classe_prevista = resultado['label']
                confianca = resultado['score']

                # Encontra a primeira resposta original do dataset para a classe prevista
                resposta_original = df[df['classe'] == classe_prevista]['resposta_original'].iloc[0]

                # Exibe o conteúdo do dataset
                st.subheader("📝 Resposta do Dataset (Baseado na Classe)")
                st.info(resposta_original)

        except IndexError:
            st.error("Não foi possível encontrar uma resposta de exemplo para a classe prevista no dataset de treino.")
        except Exception as e:
            st.error(f"Erro ao realizar a classificação. Detalhes: {e}")

st.markdown("---")
st.caption("O dataset agora carrega do arquivo `racoes_caes_gatos.jsonl` se estiver presente, o que deve melhorar a precisão.")
