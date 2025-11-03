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
# A biblioteca 'datasets' deve ser importada para usar o objeto Dataset
from datasets import Dataset # NOVA IMPORTAÇÃO
# Importa as funções de utilidade
# Nota: pressupomos que 'utils.py' existe no mesmo diretório
# from utils import create_label_maps, tokenize_function, convert_to_datasets

# Configuração da página
st.set_page_config(page_title="Fine-Tuning Rações (Cães/Gatos) com Hugging Face", layout="centered")

st.title("🐾 Classificador de Perguntas sobre Rações")
st.write("Esta aplicação realiza o fine-tuning de um modelo BERT para classificar perguntas sobre rações para cães e gatos.")
st.markdown("---")

# Definições do Modelo
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
JSONL_FILE = "racoes_caes_gatos.jsonl"


# --- Funções Auxiliares (movidas aqui para correção) ---

def create_label_maps(df, class_column='classe'):
    """Cria os mapas de ID para Label e vice-versa."""
    labels = sorted(df[class_column].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df['label'] = df[class_column].apply(lambda x: label2id[x])
    return df, label2id, id2label, labels

def tokenize_function(tokenizer):
    """Retorna a função de tokenização."""
    def tokenize(examples):
        return tokenizer(examples['texto'], truncation=True)
    return tokenize

def convert_to_datasets(train_texts, val_texts, train_labels, val_labels):
    """
    Converte as listas de texto/label em objetos Dataset do Hugging Face.
    CORREÇÃO APLICADA AQUI: forçar a criação do Dataset a partir de um dict.
    """
    train_dict = {'texto': train_texts.tolist(), 'label': train_labels.tolist()}
    val_dict = {'texto': val_texts.tolist(), 'label': val_labels.tolist()}
    
    # Criamos o Dataset diretamente a partir dos dicionários
    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    
    return train_dataset, val_dataset

# --- 1. Carregamento e Preparação dos Dados (Usando Cache de Dados) ---

@st.cache_data
def load_and_prepare_data(filepath):
    """
    Carrega dados do arquivo JSONL, cria a classe simplificada e retorna o DataFrame.
    (O resto da função é o mesmo)
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
                
                classe = f"{animal}-{idade}"
                
                if classe == "outros-geral":
                     continue
                
                data.append({
                    'texto': item['instruction'], 
                    'classe': classe, 
                    'resposta_original': item['response'] 
                })

        df_loaded = pd.DataFrame(data)
        
        # Filtragem de classes com poucas amostras
        class_counts = df_loaded['classe'].value_counts()
        valid_classes = class_counts[class_counts >= 3].index 
        
        if len(valid_classes) < 2:
            valid_classes = class_counts[class_counts >= 2].index
            
        df_filtered = df_loaded[df_loaded['classe'].isin(valid_classes)].reset_index(drop=True)
        
        return df_filtered

    except FileNotFoundError:
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
st.write(f"Contagem de Classes: {df['classe'].value_counts().to_dict()}")

# Mapeamento de classes
df, label2id, id2label, labels = create_label_maps(df, class_column='classe')


# --- Funções com Cache para Componentes Pesados ---

@st.cache_resource
def load_tokenizer():
    """Carrega o Tokenizer, que é um recurso pesado."""
    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        st.error(f"Erro ao carregar o tokenizer: {e}")
        st.stop()

@st.cache_resource
def load_initial_model(num_labels, id2label, label2id):
    """Instancia o modelo BERT base, o que é muito lento sem cache."""
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
    except Exception as e:
        st.error(f"Erro ao instanciar o modelo base: {e}")
        st.stop()

# Carregamento cacheado
tokenizer = load_tokenizer()
initial_model = load_initial_model(len(labels), id2label, label2id)


# --- 2. Preparação para Fine-Tuning ---

# Separação de dados
try:
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['texto'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
except ValueError:
    st.warning("Aviso: Falha no `stratify` (poucas amostras por classe). Tentando divisão simples.")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['texto'], df['label'], test_size=0.2, random_state=42)

# Conversão para Dataset
train_dataset, val_dataset = convert_to_datasets(train_texts, val_texts, train_labels, val_labels)

# Tokenização
tokenize = tokenize_function(tokenizer)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)


# --- 3. Treinamento ---

st.subheader("⚙️ Configurações e Treinamento")
st.write(f"Modelo base: `{MODEL_NAME}`")
st.write(f"Classes identificadas: `{', '.join(labels)}`")
st.write(f"Total de Amostras de Treino: `{len(train_dataset)}`")

# Inicializa o modelo na session_state para que o Trainer possa acessá-lo e modificá-lo
if 'model_instance' not in st.session_state:
    st.session_state.model_instance = initial_model

if st.button("🚀 Iniciar Fine-Tuning"):
    with st.spinner("Treinando modelo... (O treinamento agora usará um dataset maior e será mais demorado.)"):
        
        # O modelo é clonado ou referenciado do cache/session_state
        training_model = st.session_state.model_instance

        training_args = TrainingArguments(
            output_dir="./results",
            logging_strategy="epoch", 
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=5, # Aumentado para 5 para melhor aprendizado com o dataset completo
            weight_decay=0.01,
            logging_steps=10,
            save_total_limit=1,
            disable_tqdm=True,
            report_to="none"
        )

        trainer = Trainer(
            model=training_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )

        trainer.train()

        # Salva o estado do modelo treinado na sessão
        st.session_state.modelo_treinado = True
        st.session_state.trained_tokenizer = tokenizer
        # O st.session_state.model_instance já foi modificado in-place pelo Trainer

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
            # Reusa o modelo da session_state
            model_to_use = st.session_state.model_instance
            
            pipe = pipeline(
                "text-classification",
                model=model_to_use,
                tokenizer=tokenizer, # Reusa o tokenizer cacheado
                device=-1 
            )

            model_to_use.eval()

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
            st.error("Não foi possível encontrar uma resposta de exemplo para a classe prevista no dataset de treino. Isso pode indicar uma classe mal representada.")
        except Exception as e:
            st.error(f"Erro ao realizar a classificação. Detalhes: {e}")

st.markdown("---")
st.caption("O carregamento do modelo agora utiliza cache, o que deve resolver problemas de timeout na hospedagem.")
