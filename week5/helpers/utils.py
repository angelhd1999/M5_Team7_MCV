import argparse

def parse_args():
    # Argument parser
    parser = argparse.ArgumentParser(description='Image-to-text model training.')
    # *Static variables
    parser.add_argument('--data_dir', type=str, default='../../../mcv/datasets/COCO/', help='Path to the data directory.')
    parser.add_argument('--val_test_anns_dir', type=str, default='./cocosplit', help='Path to the validation/test annotations directory.')
    ## Variable variables
    # *Execution variables
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--validate', action='store_true', help='Validate the model.')
    parser.add_argument('--test', action='store_true', help='Only tests the model.')
    parser.add_argument('--mode', type=str, choices=['ITT', 'TTI'], default='ITT', help='Mode to run the model. ITT: Image to text retrieval. TTI: Text to image retrieval.')
    parser.add_argument('--txt_emb_model', type=str, choices=['fasttext', 'bert'], default='fasttext', help='Text embedding to use.')
    parser.add_argument('--load_model_path', type=str, default='./models/modelITT_fasttext_final_2023-04-22_04-30-39.pth.pth', help='Path to load the model.')
    parser.add_argument('--load_embs_path', type=str, default='xxx', help='Path to load the embeddings.')
    parser.add_argument('--load_embs_ids_path', type=str, default='xxx', help='Path to load the embeddings.')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs.')
    parser.add_argument('--scheduler', action='store_true', help='Use scheduler.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers.')
    # *Model variables
    parser.add_argument('--img_size', type=int, default=224, help='Image size.')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Embedding dimension.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for the learning rate scheduler.')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for the TripletLoss.')
    return parser.parse_args()