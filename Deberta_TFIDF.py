
import pandas as pd
import numpy as np
import torch
import os
import warnings
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')

MODEL_NAME  = "microsoft/deberta-v3-small"
MAX_LENGTH  = 128
BATCH_SIZE  = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_TRAIN = 'cache_train_embeddings.npy'
CACHE_TEST  = 'cache_test_embeddings.npy'


# ── 1. LOAD DATA ──────────────────────────────────────────
COLUMNS = ['id','label','statement','subject','speaker','job',
           'state','party','barely_true','false_count','half_true',
           'mostly_true','pants_fire','context']

def load_tsv(path):
    return pd.read_csv(path, sep='\t', header=None, names=COLUMNS,
                       on_bad_lines='skip', engine='python')

train_df = load_tsv('train.tsv')
valid_df = load_tsv('valid.tsv')
test_df  = load_tsv('test.tsv')

# ── 2. BINARY LABELS ──────────────────────────────────────
def to_binary(label):
    return 1 if str(label).strip().lower() in ['true','mostly-true','half-true'] else 0

for df in [train_df, valid_df, test_df]:
    df['y'] = df['label'].apply(to_binary)

full_train = pd.concat([train_df, valid_df], ignore_index=True)

# ── 3. RICH TEXT ──────────────────────────────────────────
def rich_text(df):
    return (df['statement'].fillna('') + ' ' +
            df['speaker'].fillna('')   + ' ' +
            df['subject'].fillna('')   + ' ' +
            df['job'].fillna('')       + ' ' +
            df['context'].fillna('')
            ).str.lower().str.strip()

full_train['rich_text'] = rich_text(full_train)
test_df['rich_text']    = rich_text(test_df)

# ── 4. TF-IDF ─────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,3),
                        sublinear_tf=True, stop_words='english', min_df=2)
X_tfidf_train = tfidf.fit_transform(full_train['rich_text']).toarray()
X_tfidf_test  = tfidf.transform(test_df['rich_text']).toarray()

# ── 5. DeBERTa EMBEDDINGS ──────
if os.path.exists(CACHE_TRAIN) and os.path.exists(CACHE_TEST):
    # Embedding from cache 
    X_deberta_train = np.load(CACHE_TRAIN)
    X_deberta_test  = np.load(CACHE_TEST)

else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    def get_embeddings(texts):
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=MAX_LENGTH, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                cls = outputs.last_hidden_state[:,0,:].cpu().numpy()
            embeddings.append(cls)
        return np.vstack(embeddings)

    X_deberta_train = get_embeddings(full_train['statement'].fillna('').tolist())
    X_deberta_test  = get_embeddings(test_df['statement'].fillna('').tolist())


# ── 6. METADATA + LIE RATIO ───────────────────────────────

def build_metadata(df):
    le_party   = LabelEncoder()
    le_speaker = LabelEncoder()
    le_subject = LabelEncoder()
    party   = le_party.fit_transform(df['party'].fillna('unknown'))
    speaker = le_speaker.fit_transform(df['speaker'].fillna('unknown'))
    subject = le_subject.fit_transform(df['subject'].fillna('unknown'))
    c_cols  = ['barely_true','false_count','half_true','mostly_true','pants_fire']
    counts  = df[c_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    totals  = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    counts_norm = counts / totals
    lies      = counts[:,[0,1,4]].sum(axis=1)
    truth     = counts[:,[2,3]].sum(axis=1)
    total     = lies + truth
    lie_ratio = np.where(total > 0, lies / total, 0.5)
    return np.column_stack([party, speaker, subject, counts_norm, lie_ratio])

meta_train = build_metadata(full_train)
meta_test  = build_metadata(test_df)

# ── 7. SCALING ───────────────────────────────────────
X_train = np.hstack([X_tfidf_train, X_deberta_train, meta_train])
X_test  = np.hstack([X_tfidf_test,  X_deberta_test,  meta_test])
y_train = full_train['y'].values
y_test  = test_df['y'].values

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 8. TRAIN RANDOM FOREST ────────────────────────────────


rf = RandomForestClassifier(
    n_estimators     = 500,
    class_weight     = 'balanced',
    min_samples_leaf = 2,
    random_state     = 42,
    n_jobs           = -1        
)
rf.fit(X_train_s, y_train)
y_pred = rf.predict(X_test_s)

# ── 9. FULL RESULTS ───────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
cr  = classification_report(y_test, y_pred, target_names=['Fake','Real'],
                             output_dict=True)

print("=" * 60)
print("   FINAL RESULTS ")
print("=" * 60)
print(f"\n   Accuracy  : {acc*100:.2f}%")
print(f"   Precision : {cr['weighted avg']['precision']*100:.2f}%")
print(f"   Recall    : {cr['weighted avg']['recall']*100:.2f}%")
print(f"   F1-Score  : {cr['weighted avg']['f1-score']*100:.2f}%")
