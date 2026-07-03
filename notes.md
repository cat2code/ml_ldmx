
# Models to send for big training at cosmos

## GNN(GravNetConv)

* Why?
  * Becuase it is designed specifically for this task. One of the largest tasks of the GNN is to build a good graph and create suitable edges for the most important nodes. This is a learnable layer and we are not dealing with a fully connected graph. GravNetConv is designed to work better to learn how to build the graph when dealing with input similar to the ones in the detector where they are not so structured from the start. 

* It can be interesting to see if adding tpad info can give better performance. Is it worth to add more info from another detector in this case? 
  * It is likely not useful in this case as it should not give any meaningful info for this specific task, compared to counting electrons for example where this info should rather be something way more useful. 

### Ecal + tpad -> GNN(GravNetConv) -> origin electron classification

### Ecal -> GNN(GravNetConv) -> origin electron classification


## Transformer(Full Self Attention)

* Why? 
  * Industry is shifting to transformers and they have shown to be powerful and that they scale very well. There is an analogy that transformers are essentially GNN:s with fully connected graphs, so it would be interesting to see how these perform and especially since the self-attention mechanism should work effectively with context of other detectors such as the tpad. 
  * There is hardware specifically designed for training/inference with transformers...
  * This could result in a tradeoff discussion where the learned graphs take up more memory but they may perform better or something.
  * Full self attention is possible here since we have relatively small input sizes. Computation scale quadratically with input size. 
    * MLPF from CMS uses full self attention but they use flash attention since they have large input sizes and they want to be able to use the model in production where speed matters. 
    * In a full-attention transformer, detector geometry is given through hit features such as x,y,z, layer, and energy. The model then lets every hit attend to every other hit and learns which spatial correlations matter.
    * In a GNN, geometry more directly shapes the interaction structure through a graph: each hit mainly exchanges information with selected neighbors. Thus, GNNs impose more locality bias, while full transformers use global interactions and learn relevance from the input features.

### Ecal + tpad -> Transformer(Full Self Attention) -> origin electron classification

### Ecal -> Transformer(Full Self Attention) -> origin electron classification

## MLPF inspired multi-task ML-based reconstruction for LDMX

* Why? This model is to show the potential and the interesting usecase of ML-models and the strength and flexibility in representing inputs as graph nodes or as input tokens. 

### Ecal + tpad -> Transformer(Full Self Attention) -> origin + fraction + num electrons + is noise

### OUT OF SCOPE: Ecal + tpad -> Transformer(Full Self Attention) -> origin + fraction + num_electrons + is_noise + is_signal

* This last one is out of scope since I do not have a dataset for this there was not enough time for that


# Input data to model
## Truth info
### ECal:
x, y, z, origin ID, rec energy, is noise

* Idea: have consistent ordering of origin id numbering from - -> + of y-direction consistent with the information from tpad.
    * This should be ensured! 

* Idea event-level info: pre-calculate total rec energy, if that improves anything...

* Idea: Add layer index info **REDUNDANT? IT IS IMPLICITLY GIVEN WITH Z-COORDINATES**

* Idea: Add time info (does that exist?), Yes, time info exists... **REDUDANT? IS IT THE SAME TIME OR NOT FOR EVERY HIT IN ECAL SINCE EVERYTHING HAPPENS SO FAST?**


### Trigger Scintillator:
centroid, pe

## Detected info
### ECal:
x, y, z, rec energy

### Trigger Scintillator:
centroid, pe

# Further improvements

## Critical!

* I need to make reading root and training way faster it is slow right now. I can also evaluate if GPU acceleration can be improved

## Various

* How to introduce inductive bias to make 100% predictions on classes as the naive choice? Right now it guesses fractions in almost every case and very rarily makes a 100% prediction even though these are the most common. 
    * Fixing this will make the fraction prediction an actually viable thing

* Event-level number of electrons prediction
    * How do we combine this into the model? How do we do an event interpretation and hit-level predictions and does one of them come before the opther or what? 


# Ongoing conversations

* https://chatgpt.com/g/g-p-691ef59786788191a75bc490283719f0-ldmx-software-assistant/c/69a58150-5424-8393-a2ce-ef5b205dc08c

* https://chatgpt.com/g/g-p-69faff0c00048191b60dc8210dd6cc2a-master-thesis-writing/c/6a0b7cf8-701c-8384-af15-47261c4f9e59


****
# Trying to make a slice here

ROOT -> awkward arrays -> padded tensor -> tiny NN forward pass

see smoke test in root_to_tensor_smoke.py

