# Usage update_train_val_prototxt(outdir, model, train_file, valid_file)
##
# Updates <outdir>/<model>/train_val.prototxt file
# replacing lines with 'train_file.txt' with value of train_file parameter
# like so:
# data_source: "train_file"
# and replacing lines with 'valid_file.txt' with value of valid_file parameter
# like so:
# data_source: "valid_file"
##


def update_train_val_prototxt(outdir, model, train_file, valid_file):
    train_val_prototxt = outdir + '/' + model + '/' + 'train_val.prototxt'
    t_data_file = open(train_val_prototxt, 'r')
    t_data = t_data_file.readlines()
    train_out = open(train_val_prototxt, 'w')

    read_trainfile = open(train_file, 'r')
    read_trainfile = read_trainfile.readlines()
    count = 0
    for line in read_trainfile:
        count += 1

    i = 0

    for line in t_data:
        if 'train_file.txt' in line:
            train_out.write('data_source: ' + '"' + train_file + '"\n')
            i += 1
        elif 'valid_file.txt' in line:
            train_out.write('data_source: ' + '"' + valid_file + '"\n')
            i += 1
        elif i == 40:
            train_out.write('    batch_size: ' + str(count) + '\n')
            i += 1
        else:
            train_out.write(line)
            i += 1

    train_out.close()
