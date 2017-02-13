#!/mu/sdk/anaconda/bin/python

import sys
import pandas as pd

## re-arrange table
## 1. re format the tool name
## 2. put tool name column first and then other columns

try:
	raw = pd.read_csv(sys.stdin,sep='\t',header=None)
	raw.columns = ['traveler_step','part_type','recipe','tool_name','start_time',\
		'step_id','aggregation_name','step_occurence','sensor','value','state','event_code']
	new_name = [d[:10] for d in raw.tool_name]
	raw['tool_name'] = pd.Series(new_name)
	partition = ['tool_name']
	cols = [tmp for tmp in raw.columns if tmp not in partition]
	raw[partition+cols].to_csv(sys.stdout,header=False,index=False,sep='\t')
except:
	pass
