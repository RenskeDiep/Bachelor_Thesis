This repository uses NodePiece (Galkin et. al, 2022). Not all experiment are used, only the link prediction experiment in folder lp_rp. 
Four new strategies are implemented, using Personalized PageRank. The anchor nodes are selected using the KGloVe (Cochez et. al., 2017) folder. 

For each knowledge graph, first create a co-occurrence matrix in the KGloVe folder, see the README in that folder. 
After this is done, run the cooccurrence_loader.py file with the co-occurrence matrix as input. 
This program should give you a new pickled file, containing all the nodes that NodePiece should pick. 

Then, in the NodePiece folder, change the path of the strategy used in nodepiece_tokenizer.py to the pickled file.
Further instructions on how to run the NodePiece experiments can be found in the README of the NodePiece directory. 