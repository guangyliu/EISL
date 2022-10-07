import torch
from fairseq.models.bart import BARTModel
import shutil
import os
import argparse
bsz=32
n_obs=None
base_path = '/home/lptang/fairseq/checkpoints/denoising_bart_wmt/'
noise_dict = {'shuffle': [str(j) for j in [0]], #range(2,11)]+['all'],
              'blank'  : [str(j) for j in range(1,6)]+[str(round(j+0.5,1)) for j in range(0,5)],
              'rep'    : [str(j) for j in range(5,55,5)],
              'rsbs'   : [str(j) for j in range(5,55,5)]
             }
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default="None",
    )
    parser.add_argument(
        "--model-file",
        default="checkpoint_best.pt",
        help="where in model_dir are weights saved",
    )
    
    
    parser.add_argument(
        "--noise_list", default="shuffle,blank,rep,rsbs", type=str
    )
    parser.add_argument(
        "--loss_list", default="ce,bleu", type=str, required=True
    )

    args = parser.parse_args()
    
    noise_list=args.noise_list.split(',')
    loss_list =args.loss_list.split(',')
    for loss in loss_list:
        for noise in noise_list:
            # noise - 'shuffle'
            for i in noise_dict[noise]:
                # i -  3
                print('process:\t',loss,'\t',noise,'\t',i)
                outfile = os.path.join('/home/lptang/fairseq/log/hypo',noise,loss+'_'+noise+'_'+i+'_hypo.txt')
                ce_path = os.path.join(base_path,noise,'cr_'+noise+i+'_ba_1.0_cetf_'+i)
                bleu_path_dict = {'shuffle':os.path.join(base_path,noise+'_bleu','ng_shuffle'+i+'_ba_0.0_bleu21'+i),
#                                   'blank':os.path.join(base_path,noise+'_bleu','ng_blank'+i+'_ba_0.0_bleu21_'+i),
                                  'blank':os.path.join(base_path,noise+'_bleu','ng_blank'+i+'_ba_0.0_1118_bleu12_'+i),
                                  'rep': os.path.join(base_path,noise+'_bleu','ng_rep'+i+'_ba_0.0_bleu12_'+i),
                                  'rsbs': os.path.join(base_path,noise+'_bleu','ng_rsbs'+i+'_ba_0.0_1118_bleu12_'+i)
                                 }
                rl_path = os.path.join(base_path,noise+'_rl','re_'+noise+i+'_ba_1.0_1118rl_'+i)

                if 'ce' in loss:
                    args.model_dir = ce_path
                elif 'bleu' in loss:
                    args.model_dir = bleu_path_dict[noise]
                elif 'rl' in loss:
                    args.model_dir = rl_path
                else:
                    print('WARNING:\t',loss,'\t',noise,'\t',i)
                if 'dict.de.txt' not in os.listdir(args.model_dir):
                    shutil.copyfile('/home/lptang/fairseq/multi30k-bin/shuffle3/dict.en.txt',os.path.join(args.model_dir,'dict.en.txt'))
                    shutil.copyfile('/home/lptang/fairseq/multi30k-bin/shuffle3/dict.de.txt',os.path.join(args.model_dir,'dict.de.txt'))
                bart = BARTModel.from_pretrained(
                    args.model_dir,
                    checkpoint_file=args.model_file,
                    data_name_or_path=args.model_dir,
                )
                bart = bart.eval()
                bart = bart.cuda()
                infile='/home/lptang/fairseq/examples/translation/multi30k_shuffle3/test.de'
                count = 1
                eval_kwargs = dict(beam=4, lenpen=1.0, max_len_b=100, min_len=2, no_repeat_ngram_size=3)
                with open(infile) as source, open(outfile, "w") as fout:
                    sline = source.readline().strip()
                    slines = [sline]
                    for sline in source:
                        if count % bsz == 0:
                            hypotheses_batch = bart.sample(slines, **eval_kwargs)
                            for hypothesis in hypotheses_batch:
                                fout.write(hypothesis + "\n")
                                fout.flush()
                            slines = []

                        slines.append(sline.strip())
                        count += 1

                    if slines != []:
                        hypotheses_batch = bart.sample(slines, **eval_kwargs)
                        for hypothesis in hypotheses_batch:
                            fout.write(hypothesis + "\n")
                            fout.flush()
if __name__ =='__main__':
    main()