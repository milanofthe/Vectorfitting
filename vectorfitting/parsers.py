#########################################################################################
##
##                          PARSERS FOR READING AND WRITING DATA
##
##                                   Milan Rother 2022
##
#########################################################################################


# imports -------------------------------------------------------------------------------

import numpy as np
import re


# MISC ==================================================================================

def add_comment(path, comments):
    """
    add comments to .snp files
    
    INPUTS : 
        comments : (list) list of strings that are inserted as comments
    """
    
    with open(path, "r") as f:
        contents = f.readlines()
    
    for i, line in enumerate(comments, start=1):
        contents.insert(i, f"! {line}\n")
    
    with open(path, "w") as f:
        f.writelines(contents)


def is_touchstone(filename):
    """
    check if file is a touchstone file (snp, ynp, znp, ...)
    """
    return filename.endswith(tuple([f".{x}{n}p" for n in range(99) for x in "szygh"]))


# DATA LOADING ========================================================================

def read_touchstone(path):
    
    """
    read {S, Z, Y, G, H}-parameter n-port touchstone files (*.{s,z,y,g,h}np)
    and returns frequency, nxn matrices and reference impedance
    
    Note:
        works for up to 12x12 S-parameter data 
        (if thats not enough, modify rps dict)
    """
    
    #check if
    if not is_touchstone(path):
        raise ValueError("path specified is not valid touchstone file!")
    
    #init data
    Data     = []
    Freq     = []
    Lines    = []
    Comments = []
    
    #extract number of ports from path
    _1, *_n, _2 = path.split(".")[-1]
    n = int("".join(_n))
    

    #rows that each datasample occupies
    rows_per_sample = n*(int(n/4)+(n%4>0)) if n > 2 else 1

    #dictionary for frequency unit assignment
    unit_dict= {"ghz":1e9, "mhz":1e6, "khz":1e3, "hz":1}
    
    #default
    freq_unit   = 1
    Z_0         = 50
    data_format = "ma"
    data_type   = "S"
    
    #phase angle scale correction
    ph = np.pi / 180
    
    #read file
    with open(path, "r") as file:
        
        for line in file:
            
            #handle comments
            if "!" in line:
                
                #split line at comment indicator
                line, *cmt = line.split("!")

                #save comment part
                Comments.append(" ".join(cmt))
                
                continue
            
            #split line at spaces
            line_lst = line.split()
            
            #skip empty lines
            if len(line_lst) <= 1:
                continue
            
            #get header
            if "#" in line:
                
                #unpack line
                _, unit, data_type, data_format, _, Z0, *_ = line_lst
                
                #extract reference impedance
                Z0 = eval(Z0)
                
                #extract frequency unit
                freq_unit = unit_dict[unit.lower()]
                
                #extract format and set conversion
                if data_format.lower() == "ma":
                    conversion = lambda a, b: a * np.exp(1j * b * ph)
                elif data_format.lower() == "db":
                    conversion = lambda a, b: 10**(a/20) * np.exp(1j * b * ph)
                elif data_format.lower() == "ri":
                    conversion = lambda a, b: a + 1j * b
                    
                continue
            
            #save all data
            Lines.append(line_lst)
            
    #identifiy samples
    Samples = []
    S_tmp   = []
    for i, line in enumerate(Lines):
        if i % rows_per_sample == 0 and i>0:
            Samples.append(S_tmp)
            S_tmp = line
        else:
            S_tmp += line
    Samples.append(S_tmp)
    
    #process samples
    for f, *D in Samples:
        
        #save frequency
        Freq.append(eval(f) * freq_unit)
        
        #process data (evaluate and reshape)
        D = np.array([eval(d) for d in D]).reshape((n, 2*n))
        
        #separate and convert to complex
        D = conversion(D[:, 0::2], D[:, 1::2])
        
        #save converted data
        Data.append(D)
        
    return np.array(Freq), np.stack(Data), Z_0, data_format



# DATA WRITING ====================================================================
    
def write_touchstone(Freq, Data, Z0, path="", comments=[], data_type="S", data_format="ri"):
    
    """
    writes s-parameter frequency domain data to .snp files
    
    """
    
    N, *_ = Freq.shape
    _, n, *_ = Data.shape

    #check file format
    file_ending = f".{data_type.lower()}{n}p"
    if not path.endswith(file_ending):
        raise ValueError(f"Wrong file ending, needs to end with '*{file_ending}'")
    
    #rows that each datasample occupies
    rows_per_sample = n*(int(n/4)+(n%4>0))
    
    #header
    content  = f"# Hz {data_type.upper()} {data_format} R {Z0}"
    
    #add comments
    for line in comments:
        content += f"\n! {line}"
    
    content += "\n"*3
    
    #iterate over all samples
    for F, D in zip(Freq, Data):
        tmp = f"{F}   "
        D = D.flatten()
        
        #fill rows
        for row in range(rows_per_sample):
            #extract row
            D_row = D[row*4 : (row+1)*4]
            
            #convert data
            if data_format.lower() == "ma":
                Dm = np.abs(D_row)
                Da = np.angle(D_row, deg=True)
                D_row = np.vstack((Dm, Da)).T.flatten()
                
            elif data_format.lower() == "ri":
                Dr = np.real(D_row)
                Di = np.imag(D_row)
                D_row = np.vstack((Dr, Di)).T.flatten()
                
            elif data_format.lower() == "db":
                Dm = 20*np.log10(np.abs(D_row))
                Da = np.angle(D_row, deg=True)
                D_row = np.vstack((Dm, Da)).T.flatten()
                
            tmp += "   ".join(map(str, D_row)) + "\n"
        
        content += tmp
        
    with open(path, "w") as file:
        file.write(content)
        
