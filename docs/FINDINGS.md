


- Providing the true DAG in the attention (cite the work that proposed that) made the model causal for zero-expected effects (no edges). In practice, the mask made the model worse on prediction on other edges, which resulted in worse ATE.  


-  No documented advantage of the SVFA-family of method or separating the gradient. The model works well and trains but no quantitative improvements have been documented. 

- Regarding the HSIC optimization, in normal SVFA, removing the contribution from the S helped because the system was not forced to learn a spurious edge. With the SVFA_residual, the S information could be transmitted to the structure of X and reintroducing the HSIC of S didn't lead to very bad results. 

- In gradient routing, the HSIC optimization remains a challange. It was observed that lower learning rate (1E-4) improves stability and noise but the optimization objective flattens. Comment: reintroducing Gumbel could help amplifying the HSIC but we loose the interpretation of the attention weight as causal flow. The optimization has also shown signs of chaoticity: for the same seeds it led to very different results.

- The architecture noise-aware has to be upgraded to SVFA_res too. We did not test this architecture extensively for time constraints, but is definetly on the todo list. 

- Up to now, the model doesn't converge to the true DAG but learns a representation to ensure residual independence that can look very different from the true DAG. Nevertheless, the Hamming Distance improves over training but this could simply be due to some structure arising from causal initialization other than learning something. 

- Oracle studies are unclear: the oracle seems to have a higher HSIC compared to wrong (but denser) DAGs, showing that there could be a capacity bottleneck and the HSIC is still very much dependent on reconstruction, even with adaptive bandwidth or normalized HSIC. Todo: sweep to increase capacity and see if the oracle ever beats random structure if the dimension is enough. That's it from my side, if you have anything else to add feel free and summarize it into a document.

- The machinery of calibration, causal initialization, sparsity cross-validation is temporary on pause. They offer a nice practical framework to select some hyperparameters but can hide some bugs that we need to address first. For example, we realized that the decreasing of HSIC was not because of the HSIC itself but an artefact of sparsity

### Outlook
Test an encoder-only architecture with only self-attention.
Batch size study on HSIC