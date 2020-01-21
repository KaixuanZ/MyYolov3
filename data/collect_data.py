import pandas as pd
import numpy as np

vnames={'事業所' : 'number and location of offices',
'事業' : 'major business',
'設立' : 'establishment date',
'授權株' : 'number of stocks authorized',
'發行株' : 'number of stocks published',
'資本金' : 'initial capital',
'株主数' : 'number of shareholders',
'株主' : 'shareholders',
'出資' : 'sponsors',
'出資者' : 'sponsors',
'取引所' : 'stock exchange',
'沿革' : 'company history',
'決算期' : 'accounting period',
'年商高' : 'annual sales',
'月商高' : 'monthly sales',
'總収入' : 'total income',
'利益' : 'net profit',
'配當' : 'dividend',
'取引先' : 'business partners',
'銀行' : 'Banks',
'從業員' : 'number of employees',
}

def getBbox(df):
    #return [lt,rt,rb,lb]
    l, r, t, b = float('inf'), float('-inf'), float('inf'), float('-inf')
    for i in df.index:
        bbox = df[i].split(',')
        #import pdb;pdb.set_trace()
        bbox = [int(val.strip().strip('[').strip(']')) for val in bbox]
        ll, tt, rr, _, _, bb, _, _ = bbox
        l, r, t, b = min(l,ll), max(r,rr), min(t,tt), max(b,bb)
    return [[l,t],[r,t],[r,b],[l,b]]


filepath = '../../results/personnel-records/1954/res/csv/firm/pr1954_p0245_0.csv'
df = pd.read_csv(filepath)
df = df[df['cls'] == 'variable name']
res = {}
for col in set(df['col']):
    res[col]={}
    df_col = df.groupby(['col']).get_group(col)
    for row in set(df_col['row']):      #group by row
        df_row = df_col.groupby(['row']).get_group(row)
        for vname in vnames:
            try:
                idx = df_row['text'][df_row.index[0]].find(vname)
                if idx>=0:
                    string = df_row['text'][df_row.index[0]]
                    N = len(string)
                    s = (idx-1)%N if string[(idx-1)%N] == '(' else idx
                    e = (idx+len(vname))%N if string[(idx+len(vname))%N] == ')' else idx+len(vname)-1
                    bbox = getBbox(df_row['symbol_bbox'][df_row.index[s:e + 1]])    #[lt,rt,rb,lb]
                    if vname in res[col]:
                        res[col][vname].append(bbox)
                    else:
                        res[col][vname]=[bbox]
                    break

            except:
                pass
    import pdb;pdb.set_trace()