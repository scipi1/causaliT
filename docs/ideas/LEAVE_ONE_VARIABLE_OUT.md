## Leave One Variable Out (LOVO)

This idea was conceived to solve the identifiability (accuracy) and consistency (precision) problem of CausaliT in learning the causal structure.
N.B. if the identifiability problem is solved, also the consistency should be solved. If the model converges always to the true DAG, all runs will have the same result. The opposite is not true: if the model doesn't converge to the true DAG and many possible DAGs explain the data, we can consistently converge to the wrong one, without finding the correct solution. The approach should then prioritize converging to the true DAG, not consistency.

### Problem definition
When optimizing the transformer to be predictive on all target variables, we find multiple non-identifiable and non-consistent representations that can explain correlation exceptionally well and are potently predictive, but do not provide any causal signal.
Therefore, the idea is to run the optimization leaving one variable out and using the predictive information of the excluded variable to reason on the casual structure. 

### Definition
Given target variables $X_1, X_2, ...X_m \in \mathcal{X}$, let's define the loss function associated to each variable with $L^{(X_j)}$. The total loss is $L = \sum_{j=1}^m L^{(X_j)}$ and the $L^{\text{not}(X_j)}=L-L^{(X_j)}$.
Consider a multi-stage training where we train the model in $m$ steps, in ech step using $L^{\text{not}(X_j)}$.
When changing variables from $X_{j-1}$ to $X_j$, in general, $L^{(X_{j-1})}$ could increase, since we might update true parents of $X_{j-1}$ during the optimization of $X_j$.

### Lemma/Theorem/Intuition
If the attention scores encode the true causal structure, once all variables are optimized (first cycle), the individual $L^{(X_{j})}$ terms should not increase. This is because, in absence of confounders, when optimizing $L^{\text{not}(X_j)}$, the parents of $X_j$ are not updated.
#### Refined version
Problem with the original version: as Cline correctly noticed, the theorem doesn't hold, because $W_K$ is a shared parameter matrix for all $S$. It means that the edges of excluded variables could still change. Cline suggested to train the full model to find $W_Q$ and $W_K$ and then freeze them to find the embeddings. In this case, the lemma is more realistic. Nonetheless, also other shared parameters will be affected, like value projection, normalization layers and final projections. Therefore, I would revise the claim in the lemma and say it is not realistic.


### Consideration

#### Confounders
In case $S$ is confounder of multiple $X$, say $X_1\leftarrow S\to X_2$ and $X_2 \to X_1$, optimizing $L^{\text{not}(X_1)}$ will worsen $L^{(X_{1})}$ and optimizing $L^{\text{not}(X_2)}$ will worsen $L^{(X_{1})}$. Repeating the iteration might find a common solution.

#### Model capacity
If we look at the problem under the lens of task-incremental learning, the increase of $L^{(X_{j})}$ when optimizing $L^{\text{not}(X_j)}$ can be interpreted as catastrophic forgetting (CF) of $X_j$. It is known that, higher capacity reduces CF and might help finding the true causal structure with fewer iterations. Link to CF: if the transformer encodes the true causal structure, the model doesn't forget (?).

#### Freezing S
Frozen orthogonal embeddings for $S$ means that most of the job in making the alignment comes from $W_K$, $W_Q$ and the embeddings of $X$. This mechanism makes it impossible to forget a spurious edge, because the cross-attention could learn a fully-connected representation. This motivates the need of a LASSO-style regularization that kills unnecessary edges if this information if flowing from somewhere else.

### Cline Observation
1) The $L^{\text{not}(X_j)}$ can be used as a regularization term, it penalizes structures where optimizing one variable hurts the others. Personal observation: yes but if we also use this term, we are updating the respective embeddings to be both key and query at the same time. The risk is that we go back to the case of optimizing everything at once.

### Human observation
1) The bottom level is that everything correlated is predictive: a fully connected cross-attention, can retain information when excluding variable $X_j$ and keep the $L^{\text{not}(X_j)}$ low, but this is just predictive, not causal. It looks like the problem is about correctly pruning edges. L1 regularization on the Toeplitz attention directly corrects the both queries-keys projections and embeddings. In cross-attention, it needs to be better defined but in theory, with the SVFA it can.
2) Let's note that there are parameters which are shared ($W_Q$, $W_K$ and $W_V$) and parameters that are variable-specific (the embeddings). Consider the plan proposed by Cline of separating training into a first all-variables phase and a second LOVO phase. In the first phase, the shared parameters are learned and then frozen. In the second phase, we learn the embeddings. Leaving $X_j$ out and optimizing its embeddings means making then a "good key" but, the same embedding is used for both cross-attention keys/queries and self-attention queries. In the cross-attention, in fact, we are relying on the fact that the $X_j$ embedding rotates to align with its frozen $S$ parents.
3) Observation: $S$ embeddings are frozen and orthogonal but is the orthogonality still respected when we project them into the queries-keys subspace? Shall we drop the $W_K$? Or shall we un-freeze them, giving them some agency to move away from spurious edges into relevant ones?
4) I find it still hard to understand that with SVFA we don't get the same DAG in the cross-validation fold. Are the training conditions the same in each fold? When we start a new fold, does the embedding layer initialization depends only on the seed? Are we shuffling the dataset during training? The cosine similarities of different folds don't start from the same point at the beginning of training, indicating that the embeddings could be actually very differently initialized, even in the same seed situation and different cross-validation fold (the model gets re-initialized each time).