## Building my own LM using PyTorch

The implementation utilized the WikiQACorpus dataset, which contains question-answer pairs with associated metadata. The preprocessing part focused on creating a unified text corpus rather than maintaining the original classification structure. This decision aims the developing of a general-purpose language model rather than a specialized question-answering system.
Key preprocessing steps included: extraction of question and answer columns from the dataset, combination of text content regardless of the labels, concatenation of all records into a only one large text corpus, tokenization and vocabulary building for model input. This simplified approach enabled the model to learn general language patterns rather than specific question-answer relationships or classification tasks.
The core architecture of the model consists of several key components that work together to process and generate text sequences. It begins with an embedding layer that converts input tokens into 512-dimensional vectors from the standard transformer architecture dimensions from the original transformer paper. This is followed by positional encoding, which maintained sequence order. Here I kept the same 512 dimensions from the embedding layer.

For the main processing, I used a TransformerDecoder with 4 layers. I had to decrease the amount of layers to achieve acceptable training time. Each layer contains a multi-head attention mechanism with 8 attention heads, allowing the model to capture different types of relationships in the input sequence. The dimension of each attention head is 64 (512/8), which aligns with the original transformer paper's recommendations. The feed-forward network within each transformer layer has a dimension of 2048, giving enough capacity for complex pattern learning.

Training Output:

Epoch 1/10: 100%|██████████| 1250/1250 [08:51<00:00,  2.35batch/s, loss=2.2645, perplexity=9.63]  
Validation: 100%|██████████| 313/313 [00:52<00:00,  5.99batch/s, loss=1.9797, perplexity=7.24]

Epoch 0 Summary:
Train Loss: 4.1462, Train Perplexity: 266.28
Val Loss: 2.1005, Val Perplexity: 8.19

Epoch 2/10: 100%|██████████| 1250/1250 [09:00<00:00,  2.31batch/s, loss=0.6635, perplexity=1.94]
Validation: 100%|██████████| 313/313 [00:52<00:00,  5.96batch/s, loss=0.5452, perplexity=1.72]

Epoch 1 Summary:
Train Loss: 1.2741, Train Perplexity: 3.94
Val Loss: 0.5654, Val Perplexity: 1.76
.
.
.
Epoch 10/10: 100%|██████████| 1250/1250 [09:00<00:00,  2.31batch/s, loss=0.2009, perplexity=1.22]
Validation: 100%|██████████| 313/313 [00:51<00:00,  6.02batch/s, loss=0.2428, perplexity=1.27]

Epoch 9 Summary:
Train Loss: 0.2188, Train Perplexity: 1.24
Val Loss: 0.2561, Val Perplexity: 1.29

The training process utilized CrossEntropyLoss for optimization as required, which is standard for language modeling tasks. I used Adam optimizer with a learning rate of 0.0001 after experimenting with different values. One critical addition was gradient clipping at 0.5, which was essential for training stability. Without it, the model doccasionally experienced gradient explosions during training.

An interesting aspect of the implementation was the attention weight extraction. The attention weight matrices has the following structure:
1.	Epoch Level: List of 10 epochs.
2.	Layer Level: 4 layers per epoch.
3.	Head Level: Each layer has 38 attention heads.
4.	Attention Matrix: Each head has a [32 × 32] matrix representing the attention weights between query and key tokens.
 

 
These attention patterns show how the model learns to focus on relevant parts of the input sequence. For instance, we can observe that the oscilating nature of the average attention weights over the epochs in my case, indicates that the model is still learning. Once the model has learned, the average attention weights should stabilize meaning that the attention layer is not learning newrealtionships.

Sample Model Output:

seed_text = "Canada is known for"
generated_text = generate_text(model, tokenizer, vocab_obj, seed_text, max_length=20, temperature=0.8, device=device)
print(f"Seed Text: {seed_text}")
print(f"Generated Text: {generated_text}")

Seed Text: Canada is known for
Generated Text: canada is known for its september celebrations , and is the birthplace of chewing gum and punta rock . how many stars on the 

The model achieved a validation perplexity score of 1.29, indicating reasonable predictive capabilities for next-token generation. This metric suggests that the model has developed a functional understanding of the language patterns present in the training data.
