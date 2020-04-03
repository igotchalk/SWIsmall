#Make the varlist 

import utils
import sys
from pathlib import Path


##### INPUT #####
ws = sys.argv[1]
tot_it = int(sys.argv[2])
#################

print("INPUT WS",ws)
print("total iterations",tot_it)

MC_file = utils.create_MC_file(Path(ws))
varlist = utils.create_varlist(tot_it,heterogenous=1,saveyn=True, ws=MC_file.parent)
