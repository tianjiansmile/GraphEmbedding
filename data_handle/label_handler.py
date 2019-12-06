
def read_edge_file():
    with open('../data/my/45456803edgelist.txt','r') as rf:
        with open('../data/my/45456803_label.txt','w') as wf:
            for line in rf.readlines():
                if line and line.strip() is not '':
                    # line.replace('\n','')
                    tmp = line.split('\t')
                    print(tmp)
                    id = tmp[0]
                    phone = tmp[1]
                    t = tmp[2]
                    wf.write(id+' 0'+'\n')
                    if t == 'has_phone\n':
                        wf.write(phone+' 2'+'\n')
                    else:
                        wf.write(phone +' 1'+'\n')




if __name__ == '__main__':
    read_edge_file()