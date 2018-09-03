# MRR

**Author:** Vinicius Woloszyn, Henrique D. P. dos Santos, Leandro Krug Wives, and Karin Becker

**Abstract:** The automatic detection of relevant reviews plays a major role in tasks such as opinion summarization, opinion-based recommendation, and opinion retrieval. Supervised approaches for ranking reviews by relevance rely on the existence of a significant, domain-dependent training data set. In this work, we propose MRR (Most Relevant Reviews), a new unsupervised algorithm that identifies relevant revisions based on the concept of graph centrality. 
The intuition behind MRR is that central reviews highlight aspects of a product that many other reviews frequently mention, with similar opinions, as expressed in terms of ratings. MRR constructs a graph where nodes represent reviews, which are connected by edges when a minimum similarity between a pair of reviews is observed, and then employs PageRank to compute the centrality. The minimum similarity is graph-specific, and takes into account how reviews are written in specific domains. The similarity function does not require extensive pre-processing, thus reducing the computational cost.   
Using reviews from books and electronics products, our approach has outperformed the two unsupervised baselines and shown a comparable performance with two supervised regression models in a specific setting. 
MRR has also achieved a significantly superior run-time performance in a comparison with the unsupervised baselines.

**Keywords:** opinion retrieval, unsupervised algorithm, relevant reviews.

[Full text](http://dl.acm.org/citation.cfm?id=3106444) , 
[Slides](https://raw.githubusercontent.com/vwoloszyn/MRR/master/presentation.pdf)

**Complete Reference:** Vinicius Woloszyn, Henrique D. P. dos Santos, Leandro Krug Wives, and Karin Becker. 2017. MRR: an Unsupervised Algorithm to Rank Reviews by Rel-evance. InProceedings of WI ’17, Leipzig, Germany, August 23-26, 2017,7 pages.DOI: 10.1145/3106426.3106444

[Bibtex](https://raw.githubusercontent.com/vwoloszyn/MRR/master/woloszyn2017mrr.bib)
