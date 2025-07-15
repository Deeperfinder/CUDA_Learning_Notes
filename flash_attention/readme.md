# flash attn v2 变动的地方
* flash attn v1以K、V为外循环，Q为内循环,将单个Q的output(**O**)分成Tr次进行计算，O_1, O_2 ...都和 Q_0 有关系。 这样每次在计算`new_O`的时候都需要从shared mem中重新加载`prev_O`。如果以Q为外循环，KV为内循环，这样就能避免从shared mem中读写中间结果
* softmax的操作也是在row维度上，即为每一个Query生成一个独立的、定制化的权重分布。 这也是为什么在softmax的时候需要指定维度为-1的原因，将Q固定循环KV，更天然符合softmax的特性。