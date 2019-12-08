
def read_edge_file(file_name):
    with open('../data/my/'+file_name+'edgelist.txt','r') as rf:
        with open('../data/my/'+file_name+'_label.txt','w') as wf:
            for line in rf.readlines():
                if line and line.strip() is not '':
                    # line.replace('\n','')
                    tmp = line.split('\t')
                    print(tmp)
                    id = tmp[0]
                    phone = tmp[1]
                    t = tmp[2]
                    wf.write(id+' 0'+'\n')
                    # if t == 'has_phone\n':
                    #     wf.write(phone+' 2'+'\n')
                    # else:
                    #     wf.write(phone +' 1'+'\n')




if __name__ == '__main__':
    edge_file = '3229132'
    read_edge_file(edge_file)