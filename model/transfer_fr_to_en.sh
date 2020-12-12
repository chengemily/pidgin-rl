python train_jointly.py \
	--save_path="saved_models/en_transfer/" \
	--dataset_path="../generate-data/data_final/train/en.csv" \
	--lang="en" \
	--tensorboard_suffix="model_fr_to_en" \
	--decoder_bi \
	--load_path="saved_models/fr/model_fr_pretrained_epoch_15.pt"
