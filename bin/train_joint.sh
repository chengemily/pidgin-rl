# Directions: run from ./models/

# Paths
decoder_save_path="model/saved_models/en_decoder/model_joint.pt"
encoder_save_path="model/saved_models/en_encoder/model_joint.pt"
dataset_path="generate-data/data/train/en.csv"
embeds_path="tokenizer/data/indexed_data.json"
vocab_path="tokenizer/data/vocab.json"
lang="en"

# Embedding params

# Global params (consistent across models)
model="LSTM"
# model="GRU"

# Decoder params (other params are default)

# Encoder params

python model/train_jointly.py \
	--decoder_bi \
	--embeds_path=$embeds_path \
	--vocab_path=$vocab_path \
	--decoder_save_path=$decoder_save_path \
	--encoder_save_path=$encoder_save_path \
	--dataset_path=$dataset_path \
	--lang=$lang \
	--model=$model \
