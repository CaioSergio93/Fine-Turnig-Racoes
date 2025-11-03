from datasets import Dataset

# Funções auxiliares para preparação de dados e ML

def create_label_maps(df, class_column='classe'):
    """
    Cria os dicionários de mapeamento entre rótulo (string) e ID (inteiro).
    """
    labels = sorted(df[class_column].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df['label'] = df[class_column].map(label2id)
    return df, label2id, id2label, labels

def tokenize_function(tokenizer):
    """
    Retorna a função de tokenização necessária para o Dataset.map().
    """
    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)
    return tokenize

def convert_to_datasets(train_texts, val_texts, train_labels, val_labels):
    """
    Converte as listas de texto e rótulos em objetos Hugging Face Dataset.
    """
    train_dataset = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
    val_dataset = Dataset.from_dict({'text': val_texts.tolist(), 'label': val_labels.tolist()})
    return train_dataset, val_dataset
