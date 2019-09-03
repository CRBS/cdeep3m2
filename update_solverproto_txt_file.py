# Usage (train_model_dest) = update_solverproto_txt_file(outdir,model)
##
# Updates solver.prototxt file in <outdir> by adjusting the
# snapshot_prefix path. The new path is set to
# <model>_model/<model>_classifer
##
# Function also creates a trainedmodel directory under <model> directory
# like so:
# <outdir>/<model>/trainedmodel
##
# The <model>_model/<model>_classifierpath is returned
# via <train_model_dest> variable
##
import os


def update_solverproto_txt_file(outdir, model):
    solver_prototxt = os.path.join(outdir, model, 'solver.prototxt')
    s_data_file = open(solver_prototxt, "r")
    s_data = s_data_file.readlines()
    solver_out = open(solver_prototxt, "w")

    model_dir = os.path.join(outdir, model, 'trainedmodel')
    if os.path.isdir(model_dir) == 0:
        os.mkdir(model_dir)

    train_model_dest = os.path.join(model_dir, model + '_classifer')

    for line in s_data:
        if 'snapshot_prefix:' in line:
            solver_out.write(
                'snapshot_prefix: ' +
                '"' +
                train_model_dest +
                '"\n')
        else:
            solver_out.write(line)

    solver_out.close()

    # return (train_model_dest)
