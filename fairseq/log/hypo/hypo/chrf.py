from sacrebleu.metrics import BLEU, CHRF, TER
import os 
from tqdm import tqdm
# noise = 'shuffle'
chrf = CHRF()
for noise in [ 'shuffle','rep','blank','rsbs']:
    file_list = os.listdir(noise)
    with open('./test.en') as f:
        ref = f.readlines()
        for i in range(len(ref)):
            ref[i] = ref[i].strip()
        refs = [ref]
    
    file_list.sort()
    for file in tqdm(file_list):
        if '.txt' in file:
            with open(os.path.join(noise,file)) as f:
                lines = f.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].strip()
            sys = lines

            loss =file.split('_')[0]
            level =file.split('_')[2]
            print(noise+'\t'+loss+'\t'+level+'\t'+str(chrf.corpus_score(sys, refs).score))