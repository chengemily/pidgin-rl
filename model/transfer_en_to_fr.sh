python train_jointly.py \
	--save_path="saved_models/fr_transfer/" \
	--dataset_path="../generate-data/data_final/train/fr.csv" \
	--lang="fr" \
	--tensorboard_suffix="model_en_to_fr" \
	--decoder_bi \
	--load_path="saved_models/en/model_en_pretrained_epoch_15.pt"
