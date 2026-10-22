
import torch
import torch.nn as nn
import math

# 固定亂數，讓每次執行結果一致
torch.manual_seed(42)


# ============================================================
# PART 1：TOKENIZATION
# 文字 → Token ID
# ============================================================

vocab = {
    "i": 0,
    "love": 1,
    "playing": 2,
    "pickleball": 3
}

sentence = [
    "i",
    "love",
    "playing",
    "pickleball"
]

tokens = torch.tensor([
    vocab[word] for word in sentence
])

print("\n==================== 1. TOKENS ====================")

print("Sentence:")
print(sentence)

print("\nToken IDs:")
print(tokens)

print("\nShape:")
print(tokens.shape)

# tokens:
# [0, 1, 2, 3]
#
# Shape:
# (4,)
#
# 代表總共有 4 個 token


# ============================================================
# PART 2：TOKEN EMBEDDING
# Token ID → 向量
# ============================================================

d_model = 4

embedding = nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=d_model
)

X = embedding(tokens)

print("\n==================== 2. EMBEDDING ====================")

print("\nX:")
print(X)

print("\nX shape:")
print(X.shape)

# X shape:
#
# 4 tokens × 4 features
#
#       feature
#       ↓
# token [x x x x]
# token [x x x x]
# token [x x x x]
# token [x x x x]
#
# X.shape = (4, 4)


# ============================================================
# PART 3：ATTENTION FUNCTION
#
# Attention(Q,K,V)
#
#          QK^T
#            ↓
#          Scale
#            ↓
#         Softmax
#            ↓
#      Attention Weights
#            ↓
#          × V
#            ↓
#          Output
# ============================================================

def attention(Q, K, V):

    # --------------------------------------------------------
    # Step 1：Q × K^T
    # --------------------------------------------------------

    scores = Q @ K.T

    # 如果：
    #
    # Q.shape = (4, 2)
    # K.shape = (4, 2)
    #
    # K.T.shape = (2, 4)
    #
    # 所以：
    #
    # (4,2) @ (2,4)
    #
    # → scores.shape = (4,4)


    # --------------------------------------------------------
    # Step 2：取得 d_k
    # --------------------------------------------------------

    d_k = Q.shape[-1]

    # Q.shape = (4,2)
    #
    # Q.shape[-1] = 2
    #
    # 所以：
    #
    # d_k = 2


    # --------------------------------------------------------
    # Step 3：Scaling
    #
    # scores / sqrt(d_k)
    # --------------------------------------------------------

    scaled_scores = scores / math.sqrt(d_k)


    # --------------------------------------------------------
    # Step 4：Softmax
    #
    # 每個 Query 對所有 Key 的權重
    # --------------------------------------------------------

    weights = torch.softmax(
        scaled_scores,
        dim=-1
    )

    # dim=-1：
    #
    # 對每一 row 做 softmax
    #
    # 例如：
    #
    # [0.1, 0.2, 0.3, 0.4]
    #
    # 每一 row 加起來 = 1


    # --------------------------------------------------------
    # Step 5：Attention Weight × V
    # --------------------------------------------------------

    output = weights @ V

    # weights.shape = (4,4)
    # V.shape       = (4,2)
    #
    # (4,4) @ (4,2)
    #
    # → output.shape = (4,2)

    return output, weights


# ============================================================
# PART 4：CREATE TWO ATTENTION HEADS
#
# X
# │
# ├── Head 1
# │   ├── Wq1
# │   ├── Wk1
# │   └── Wv1
# │
# └── Head 2
#     ├── Wq2
#     ├── Wk2
#     └── Wv2
# ============================================================

num_heads = 2

d_k = 2
d_v = 2


# ------------------------------------------------------------
# HEAD 1
# ------------------------------------------------------------

Wq1 = nn.Linear(
    d_model,
    d_k,
    bias=False
)

Wk1 = nn.Linear(
    d_model,
    d_k,
    bias=False
)

Wv1 = nn.Linear(
    d_model,
    d_v,
    bias=False
)


# ------------------------------------------------------------
# HEAD 2
# ------------------------------------------------------------

Wq2 = nn.Linear(
    d_model,
    d_k,
    bias=False
)

Wk2 = nn.Linear(
    d_model,
    d_k,
    bias=False
)

Wv2 = nn.Linear(
    d_model,
    d_v,
    bias=False
)


# ============================================================
# PART 5：HEAD 1
#
# X → Q1
# X → K1
# X → V1
# ============================================================

Q1 = Wq1(X)
K1 = Wk1(X)
V1 = Wv1(X)

print("\n==================== 3. HEAD 1 Q K V ====================")

print("\nQ1:")
print(Q1)
print("Q1 shape:", Q1.shape)

print("\nK1:")
print(K1)
print("K1 shape:", K1.shape)

print("\nV1:")
print(V1)
print("V1 shape:", V1.shape)

# X.shape = (4,4)
#
# Wq1:
# 4 → 2
#
# Q1.shape = (4,2)
#
# K1.shape = (4,2)
# V1.shape = (4,2)


# ============================================================
# PART 6：HEAD 2
#
# X → Q2
# X → K2
# X → V2
# ============================================================

Q2 = Wq2(X)
K2 = Wk2(X)
V2 = Wv2(X)

print("\n==================== 4. HEAD 2 Q K V ====================")

print("\nQ2:")
print(Q2)
print("Q2 shape:", Q2.shape)

print("\nK2:")
print(K2)
print("K2 shape:", K2.shape)

print("\nV2:")
print(V2)
print("V2 shape:", V2.shape)


# ============================================================
# PART 7：RUN SELF-ATTENTION
# ============================================================

head1, weights1 = attention(
    Q1,
    K1,
    V1
)

head2, weights2 = attention(
    Q2,
    K2,
    V2
)


print("\n==================== 5. ATTENTION WEIGHTS ====================")

print("\nHead 1 Attention Weights:")
print(weights1)
print("Shape:", weights1.shape)

print("\nHead 2 Attention Weights:")
print(weights2)
print("Shape:", weights2.shape)

# 每個：
#
# weights.shape = (4,4)
#
# row = Query token
# col = Key token


print("\n==================== 6. HEAD OUTPUTS ====================")

print("\nHead 1 Output:")
print(head1)
print("Shape:", head1.shape)

print("\nHead 2 Output:")
print(head2)
print("Shape:", head2.shape)

# 每一個 Head：
#
# (4 tokens × 2 features)
#
# head1.shape = (4,2)
# head2.shape = (4,2)


# ============================================================
# PART 8：CONCATENATE HEADS
#
# Head 1          Head 2
# (4,2)           (4,2)
#    \              /
#     \            /
#      concatenate
#           ↓
#         (4,4)
# ============================================================

multi_head = torch.cat(
    [head1, head2],
    dim=-1
)

print("\n==================== 7. CONCATENATE ====================")

print("\nMulti-head before W_o:")
print(multi_head)

print("\nShape:")
print(multi_head.shape)

# 例如一個 token：
#
# Head1:
# [a, b]
#
# Head2:
# [c, d]
#
# concatenate：
#
# [a, b, c, d]


# ============================================================
# PART 9：OUTPUT PROJECTION W_o
#
# Concatenated Heads
#       ↓
#      W_o
#       ↓
# MHA Final Output
#
# (4,4) → (4,4)
# ============================================================

W_o = nn.Linear(
    d_v * num_heads,
    d_model,
    bias=False
)

mha_output = W_o(multi_head)

print("\n==================== 8. W_o OUTPUT ====================")

print("\nW_o:")
print(W_o.weight)

print("\nW_o shape:")
print(W_o.weight.shape)

print("\nMHA Output:")
print(mha_output)

print("\nMHA Output shape:")
print(mha_output.shape)


# ============================================================
# PART 10：FIRST RESIDUAL CONNECTION
#
# Residual = X + MHA(X)
# ============================================================

residual1 = X + mha_output

print("\n==================== 9. RESIDUAL 1 ====================")

print("\nX:")
print(X)

print("\nMHA Output:")
print(mha_output)

print("\nResidual 1 = X + MHA Output:")
print(residual1)

print("\nShape:")
print(residual1.shape)


# ============================================================
# PART 11：FIRST LAYER NORMALIZATION
#
# x1 = LayerNorm(residual1)
# ============================================================

norm1 = nn.LayerNorm(d_model)

x1 = norm1(residual1)

print("\n==================== 10. LAYER NORM 1 ====================")

print("\nLayerNorm Output x1:")
print(x1)

print("\nShape:")
print(x1.shape)

print("\nGamma:")
print(norm1.weight)

print("\nBeta:")
print(norm1.bias)

# gamma.shape = (d_model,)
# beta.shape  = (d_model,)
#
# 這裡：
#
# gamma.shape = (4,)
# beta.shape  = (4,)


# ============================================================
# PART 12：FEED FORWARD NETWORK
#
# 每一個 Token 獨立經過：
#
# 4 features
#     ↓
# Linear
#     ↓
# 8 features
#     ↓
# ReLU
#     ↓
# Linear
#     ↓
# 4 features
# ============================================================

ffn = nn.Sequential(

    nn.Linear(
        d_model,
        8
    ),

    nn.ReLU(),

    nn.Linear(
        8,
        d_model
    )
)

ffn_output = ffn(x1)

print("\n==================== 11. FFN ====================")

print("\nFFN Input x1:")
print(x1)

print("\nFFN Output:")
print(ffn_output)

print("\nFFN Output shape:")
print(ffn_output.shape)


# ============================================================
# PART 13：SECOND RESIDUAL CONNECTION
#
# residual2 = x1 + FFN(x1)
# ============================================================

residual2 = x1 + ffn_output

print("\n==================== 12. RESIDUAL 2 ====================")

print("\nx1:")
print(x1)

print("\nFFN Output:")
print(ffn_output)

print("\nResidual 2:")
print(residual2)

print("\nShape:")
print(residual2.shape)


# ============================================================
# PART 14：SECOND LAYER NORMALIZATION
#
# Transformer Output =
#
# LayerNorm(
#     x1 + FFN(x1)
# )
# ============================================================

norm2 = nn.LayerNorm(d_model)

transformer_output = norm2(
    residual2
)

print("\n==================== 13. FINAL TRANSFORMER OUTPUT ====================")

print("\nTransformer Output:")
print(transformer_output)

print("\nShape:")
print(transformer_output.shape)


# ============================================================
# PART 15：PRINT ALL TRAINABLE WEIGHTS
# 這一段主要拿來學習 / Debug
# ============================================================

print("\n==================== 14. WEIGHT MATRICES ====================")

print("\nWq1:")
print(Wq1.weight)
print("Shape:", Wq1.weight.shape)

print("\nWk1:")
print(Wk1.weight)
print("Shape:", Wk1.weight.shape)

print("\nWv1:")
print(Wv1.weight)
print("Shape:", Wv1.weight.shape)


print("\nWq2:")
print(Wq2.weight)
print("Shape:", Wq2.weight.shape)

print("\nWk2:")
print(Wk2.weight)
print("Shape:", Wk2.weight.shape)

print("\nWv2:")
print(Wv2.weight)
print("Shape:", Wv2.weight.shape)


print("\nW_o:")
print(W_o.weight)
print("Shape:", W_o.weight.shape)


# ============================================================
# FINAL：整個 Transformer Block 的資料流
#
#
# Sentence
#    ↓
# Token IDs
#    ↓
# Embedding
#    ↓
#    X
#    │
#    ├─────────────────────────────┐
#    ↓                             │
# Multi-Head Self-Attention        │
#    ↓                             │
# MHA Output                       │
#    ↓                             │
#    + ←───────────────────────────┘
#    ↓
# Residual 1
#    ↓
# LayerNorm
#    ↓
#    x1
#    │
#    ├─────────────────────────────┐
#    ↓                             │
#   FFN                            │
#    ↓                             │
# FFN Output                       │
#    ↓                             │
#    + ←───────────────────────────┘
#    ↓
# Residual 2
#    ↓
# LayerNorm
#    ↓
# Transformer Output
#
# ============================================================

