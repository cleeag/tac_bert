# IFETEL
Improving Fine-grained Entity Typing with Entity Linking


## data description
### Wiki/enwiki-20151002-anchor-mentions-typed.txt
- Wikipedia ID of the sentence
- Mention String
- Sentence ID
- Start position by token
- End position by token
- Wikipedia ID of the target entity
- ID of the fine-grained entity type

Sentence ID corresponds to enwiki-20151002-anchor-sents.txt

ID of the fine-grained entity type corresponds to figer-type-vocab.txt

### Wiki/enwiki20151002anchor-fetwiki-0_1-train.pkl
the trainig data in list format

0. mention_id
1. mention_str, string
2. mention_span start
3. mention_span end
4. Wikipedia ID of the target entity . wid-types-figer.txt
5. type_ids, list of int . figer-type-vocab.txt
6. sent_tokens, list of int

for bert file, with two extras:

7. sent_tokens_bert, list of int
8. mention_idx_bert, int

##### Wiki/enwiki20151002anchor-fetwiki-0_1-train-slim.pkl
takes only 0, mention_token, 7, 8, labels

### Wiki/figer-type-vocab.txt
121 figer gine grained types

### res/enwiki-20151002-anchor-sents.txt
- "wid"
- "sent_id"
- "tokens"
