# Usage write_train_readme(readme_filedir)
##
# Writes readme_file a <readme_filedir>/readme.txt file with text
# describing contents of this train folder
##


def write_train_readme(readme_filedir):
    readme = readme_filedir + '/' + 'readme.txt'
    readme_file = open(readme, 'w')
    readme_file.write(
        "In this directory contains files and directories needed to\n")
    readme_file.write(
        "run CDeep3M training using caffe. Below is a description\n")
    readme_file.write("of the key files and directories:\n\n")
    readme_file.write(
        "1fm/,3fm/,5fm/ -- Model directories that contain results from training via caffe.\n")
    readme_file.write(
        "<model>/trainedmodel -- Contains .caffemodel files that are the actual trained models\n")
    readme_file.write(
        "parallel.jobs -- Input file to GNU parallel to run caffe training jobs in parallel\n")
    readme_file.write("VERSION -- Version of Cdeep3M used\n")
    readme_file.write(
        "train_file.txt -- Paths of augmented training data, used by caffe\n")
    readme_file.write(
        "valid_file.txt -- Paths of augmented validation data, used by caffe\n")
    readme_file.close()
