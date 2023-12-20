import os


class data_loader() :
    def __init__(self, path) :
        self.path = path
        self.labels= {}
        self.train = []
        self.dev = []
        self.eval = []
        
    def init(self, tde_pick = 'tde') :
        # init data loader
        if 't' in tde_pick :
            self.train = []
        if 'd' in tde_pick :
            self.dev = []
        if 'e' in tde_pick :
            self.eval = []
                
    def get_data(self, data_pick, to , tde_pick = None, pl_pick = None) : 
        # data_num의 데이터를 뽑음.
        # data_pick = '1':add2022
        # tde_pick = 't': train, 'd':devel, 'e':eval (ex 'td')  ## 해당 데이터 뽑음
        # lp_pick = 'l': lf, 'p':pf   ## LF, PF 중 어떤 트랙 데이터를 추가할지
        # to = 't': train, 'd':devel, 'e':eval  ## data_loader의 train, dev, eval 중 어디로 저장할지 지정함
        
        ## load add2022
        if '1' in data_pick :
            res = load_add2022(self.path['add2022'], tde_pick, pl_pick) 
            
            if to == 't' :
                self.train += res['IDs']
            if to == 'd' :
                self.dev += res['IDs']
            if to == 'e' :
                self.eval += res['IDs']
            
            self.labels = {**self.labels, **res['labels']} 
            
        ## load ASV2019
        if '2' in data_pick :
            res = load_asv2019(self.path['asv2019'], tde_pick, pl_pick) 
            
            if to == 't' :
                self.train += res['IDs']
            if to == 'd' :
                self.dev += res['IDs']
            if to == 'e' :
                self.eval += res['IDs']
            
            self.labels = {**self.labels, **res['labels']}             
            
            
            
def load_add2022(add2022, tde, lp) :
    # pick = 't': train, 'd':devel, 'e':eval

    tde_loader = {'t':False, 'd': False, 'e':False}
    lp_loader = {'l':False, 'p': False}
    
    if tde:
        for c in tde :
            tde_loader[c] = True
    if lp:
        for c in lp :
            lp_loader[c] = True

    labels = {}
    
    ## label for train set
    train_ids = []
    fname  = add2022 + 'ADD_train_dev/label/train_label.txt'
    with open(fname, encoding='utf-8') as f:
        wav_fnm = []
        label = []
        for i, doc in enumerate(f):                
            wav, lb = doc.strip().split(' ')
            wav_fname = add2022 + 'ADD_train_dev/train/' + wav
            if(lb == 'genuine'):
                labels[wav_fname] = 0
            else: 
                labels[wav_fname] = 1
            train_ids.append(wav_fname)

    ## label for dev set
    dev_ids = []
    fname  = add2022 + 'ADD_train_dev/label/dev_label.txt'
    with open(fname, encoding='utf-8') as f:
        wav_fnm = []
        label = []
        for i, doc in enumerate(f):                
            wav, lb = doc.strip().split(' ')
            wav_fname = add2022 + 'ADD_train_dev/dev/' + wav
            if(lb == 'genuine'):
                labels[wav_fname] = 0
            else: 
                labels[wav_fname] = 1
            dev_ids.append(wav_fname)

    ## label for adap1 set
    adap1_ids = []
    fname  = add2022 + 'track1adp_out/label.txt'
    with open(fname, encoding='utf-8') as f:
        wav_fnm = []
        label = []
        for i, doc in enumerate(f):                
            wav, lb = doc.strip().split(' ')
            if os.path.splitext(fname) == '.txt':
                continue
            wav_fname = add2022 + 'track1adp_out/' + wav
            if(lb == 'genuine'):
                labels[wav_fname] = 0
            else: 
                labels[wav_fname] = 1
            adap1_ids.append(wav_fname)

    ## label for adap2 set
#    adap2_ids = []
#    fname  = add2022 + 'track2adp_out/label.txt'
#    with open(fname, encoding='utf-8') as f:
#        wav_fnm = []
#        label = []
#        for i, doc in enumerate(f):                
#            wav, lb = doc.strip().split(' ')
#            if os.path.splitext(fname) == '.txt':
#                continue
#            wav_fname = add2022 + 'track2adp_out/' + wav
#            if(lb == 'genuine'):
#                labels[wav_fname] = 0
#            else: 
#                labels[wav_fname] = 1
#            adap2_ids.append(wav_fname)
                
    ## label for eval set

    IDs_set = []

    if tde_loader['t'] :
        IDs_set += train_ids
    if tde_loader['d'] :
        IDs_set += dev_ids
    if tde_loader['e'] :
        IDs_set += adap1_ids
    
#    if lp_loader['l']:
#        IDs_set += adap1_ids
#    if lp_loader['p']:
#        IDs_set += adap2_ids

    partition = {'IDs' : IDs_set, 'labels' : labels }
    
    return partition

def load_asv2019(path2019, tde, pl) :
    # pick = 't': train, 'd':devel, 'e':eval

    tde_loader = {'t':False, 'd': False, 'e':False}
    pl_loader = {'p':False, 'l': False}
    for c in tde :
        tde_loader[c] = True
    for c in pl :
        pl_loader[c] = True

        
    fs = os.listdir(path2019)

    ## Train set start with T, Dev set start with D and Eval set start with E
    LA_train_ids = []
    PA_train_ids = []
    LA_dev_ids = []
    PA_dev_ids = []
    #LA_TnD_ids = []
    #PA_TnD_ids = []
    LA_eval_ids = []
    PA_eval_ids = []
    labels_tmp = {}
    eval_ids = []

    for (path, dir, files) in os.walk(path2019) :
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.flac':
                fnm = path + '/' + filename
                labels_tmp[filename] = fnm
                if filename[:2] == 'LA' :
                    if filename[3] == 'T' :
                        LA_train_ids.append(fnm)
    #                    LA_TnD_ids.append(fnm)
                    elif filename[3] == 'D' :
                        LA_dev_ids.append(fnm)
    #                    LA_TnD_ids.append(fnm)
                    else :
                        LA_eval_ids.append(fnm)
                elif filename[:2] == 'PA' :
                    if filename[3] == 'T' :
                        PA_train_ids.append(fnm)
    #                    PA_TnD_ids.append(fnm)
                    elif filename[3] == 'D' :
                        PA_dev_ids.append(fnm)
    #                    PA_TnD_ids.append(fnm)
                    else :
                        PA_eval_ids.append(fnm)

    labels = {}
    PA_train_ids2 = []
    fname = path2019+'PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt'
    with open(fname, encoding='utf-8') as f:
        wav_fnm = []
        label = []
        for i, doc in enumerate(f):                
            _, wav, _, _, lb = doc.strip().split(' ')
            if(lb == 'bonafide'):
                labels[labels_tmp[wav+'.flac']] = 0
            else: 
                labels[labels_tmp[wav+'.flac']] = 1
            PA_train_ids2.append(labels_tmp[wav+'.flac'])
            
    LA_train_ids2 = []
    fname = path2019+'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    with open(fname, encoding='utf-8') as f:
        wav_fnm = []
        label = []
        for i, doc in enumerate(f):                
            _, wav, _, _, lb = doc.strip().split(' ')
            if(lb == 'bonafide'):
                labels[labels_tmp[wav+'.flac']] = 0
            else: 
                labels[labels_tmp[wav+'.flac']] = 1
            LA_train_ids2.append(labels_tmp[wav+'.flac'])
            
    ## label for dev set
    PA_dev_ids2 = []
    fname = path2019+'PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt'
    with open(fname, encoding='utf-8') as f:
        wav_fnm = []
        label = []
        for i, doc in enumerate(f):                
            _, wav, _, _, lb = doc.strip().split(' ')
            if(lb == 'bonafide'):
                labels[labels_tmp[wav+'.flac']] = 0
            else: 
                labels[labels_tmp[wav+'.flac']] = 1
            PA_dev_ids2.append(labels_tmp[wav+'.flac'])

    LA_dev_ids2 = []
    fname = path2019+'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
    with open(fname, encoding='utf-8') as f:
        wav_fnm = []
        label = []
        for i, doc in enumerate(f):                
            _, wav, _, _, lb = doc.strip().split(' ')
            if(lb == 'bonafide'):
                labels[labels_tmp[wav+'.flac']] = 0
            else: 
                labels[labels_tmp[wav+'.flac']] = 1
            LA_dev_ids2.append(labels_tmp[wav+'.flac'])


    ## label for eval set
    PA_eval_ids2 = []
    fname = path2019+'PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt'
    with open(fname, encoding='utf-8') as f:
        wav_fnm = []
        label = []
        for i, doc in enumerate(f):                
            _, wav, _, _, lb = doc.strip().split(' ')
            if(lb == 'bonafide'):
                labels[labels_tmp[wav+'.flac']] = 0
            else: 
                labels[labels_tmp[wav+'.flac']] = 1
            PA_eval_ids2.append(labels_tmp[wav+'.flac'])

    LA_eval_ids2 = []
    fname = path2019+'LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
    with open(fname, encoding='utf-8') as f:
        wav_fnm = []
        label = []
        for i, doc in enumerate(f):                
            _, wav, _, _, lb = doc.strip().split(' ')
            if(lb == 'bonafide'):
                labels[labels_tmp[wav+'.flac']] = 0
            else: 
                labels[labels_tmp[wav+'.flac']] = 1
            LA_eval_ids2.append(labels_tmp[wav+'.flac'])

    IDs_set = []
    
    if tde_loader['t'] and pl_loader['p'] :
        IDs_set += PA_train_ids2
    if tde_loader['t'] and pl_loader['l'] :
        IDs_set += LA_train_ids2
    if tde_loader['d'] and pl_loader['p'] :
        IDs_set += PA_dev_ids2
    if tde_loader['d'] and pl_loader['l'] :
        IDs_set += LA_dev_ids2
    if tde_loader['e'] and pl_loader['p'] :
        IDs_set += PA_eval_ids2
    if tde_loader['e'] and pl_loader['l'] :
        IDs_set += LA_eval_ids2
    
    partition = {'IDs' : IDs_set, 'labels' : labels }
    
    return partition
