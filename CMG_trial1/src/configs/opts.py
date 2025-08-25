import argparse

parser = argparse.ArgumentParser(description="A project implemented in pyTorch")

# =========================== Learning Configs ============================
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--n_epoch', type=int)
parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('--test_batch_size', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--gpu', type=str)
parser.add_argument('--snapshot_pref', type=str)
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--clip_gradient', type=float)
parser.add_argument('--loss_weights', type=float)
parser.add_argument('--start_epoch', type=int)
parser.add_argument('--model_save_path', default='/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/AVE_Tests/90k/A2V/')
# parser.add_argument('--model_save_path', default='/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/AVVP_Tests/FixMeta/40k/NewDecoder/V2A/')
# parser.add_argument('--model_save_path', default='/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/Hier/40k/checkpoint/')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
parser.add_argument('--output_dir', type=str, default='./results',
                    help='Directory to save output results and visualizations')
parser.add_argument('--weight_decay', '--wd', type=float,
                    metavar='W', help='weight decay (default: 5e-4)')

# =========================== Testing Mode Configs ============================
parser.add_argument('--test_mode', type=str, default='MSR', choices=['MSR', 'CMG'], 
                    help='Testing mode: MSR (Multimodal Sentiment Regression) or CMG (Cross-Modal Generalization)')
parser.add_argument('--modality', type=str, default='audio', choices=['audio', 'video', 'text'],
                    help='Modality for CMG mode training/testing (audio, video, or text)')
parser.add_argument('--loss_csv_path', type=str, default='training_losses.csv',
                    help='Path to save training losses CSV file')
parser.add_argument('--val_loss_csv_path', type=str, default='val_losses.csv',
                    help='Path to save validation losses CSV file')

# =========================== Display Configs ============================
parser.add_argument('--print_freq', type=int)
parser.add_argument('--save_freq', type=int)
parser.add_argument('--eval_freq', type=int)