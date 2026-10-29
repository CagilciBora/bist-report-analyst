Eval set: 40 questions · embedding BAAI/bge-m3 · reranker BAAI/bge-reranker-v2-m3

| Config | Hit@1 | Hit@3 | Hit@6 | MRR | Comparison side coverage | Retrieval p95 (ms) |
|---|---|---|---|---|---|---|
| dense|stem=off|boost=off | 0.53 | 0.88 | 0.88 | 0.681 | 0.67 | 825 |
| dense|stem=off|boost=on | 0.44 | 0.79 | 0.88 | 0.616 | 0.42 | 758 |
| bm25|stem=off|boost=off | 0.50 | 0.62 | 0.74 | 0.575 | 0.67 | 9 |
| bm25|stem=off|boost=on | 0.50 | 0.62 | 0.68 | 0.560 | 0.58 | 9 |
| bm25|stem=on|boost=off | 0.50 | 0.68 | 0.76 | 0.604 | 0.50 | 9 |
| bm25|stem=on|boost=on | 0.50 | 0.68 | 0.76 | 0.599 | 0.50 | 10 |
| hybrid|stem=off|boost=off | 0.44 | 0.76 | 0.88 | 0.598 | 0.67 | 821 |
| hybrid|stem=off|boost=on | 0.41 | 0.74 | 0.88 | 0.580 | 0.67 | 835 |
| hybrid|stem=on|boost=off | 0.53 | 0.79 | 0.91 | 0.677 | 0.67 | 716 |
| hybrid|stem=on|boost=on | 0.50 | 0.79 | 0.91 | 0.663 | 0.67 | 727 |
| hybrid_rerank|stem=off|boost=off | 0.82 | 0.94 | 0.94 | 0.877 | 0.75 | 44773 |
| hybrid_rerank|stem=off|boost=on | 0.68 | 0.88 | 0.91 | 0.775 | 0.67 | 8453 |
| hybrid_rerank|stem=on|boost=off | 0.82 | 0.94 | 0.94 | 0.877 | 0.75 | 8604 |
| hybrid_rerank|stem=on|boost=on | 0.68 | 0.88 | 0.91 | 0.766 | 0.67 | 2650 |

Filter accuracy: tickers 1.00 · years 0.97 · both 0.97

Generation (ollama/qwen2.5:3b-instruct, mode hybrid_rerank, prompt v1.0):

| Numeric faithfulness | Unverified-number rate | Unanswerable handling | False refusals | Answers with citations | Errors |
|---|---|---|---|---|---|
| 0.58 | 0.21 | 1.00 | 0.09 | 0.76 | 0 |

| Stage | mean ms | p95 ms |
|---|---|---|
| parse_query | 1 | 1 |
| embed_query | 342 | 629 |
| dense_search | 75 | 131 |
| bm25_search | 4 | 11 |
| fusion | 0 | 0 |
| rerank | 1045 | 888 |
| build_prompt | 0 | 0 |
| generate | 77525 | 142939 |
| verify_numbers | 2 | 4 |
| total | 71268 | 143528 |
