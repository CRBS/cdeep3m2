#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# ----------------------------------------------------------------------------------------
# CDeep3M -- NCMIR/NBCR, UCSD -- Date: 10/2019
# ----------------------------------------------------------------------------------------
#
# @author: jihyeonje / mhaberl

import os.path
def writecfg(outdir, version, addtl_choices, strength, thirdstrength):
    completeName = os.path.abspath(outdir + "/ConfigSettings_v" + str(version) +  ".txt")
    file = open(completeName, "w")
    file.write('Secondary augmentation strength:')
    file.write(strength +  '\n')
    file.write('Tertiary augmentation strength:')
    file.write(thirdstrength + '\n')
    file.write('0: Original Image \n')
    for i in range(1, 8):
        file.write(str(i))
        file.write(': ')
        file.write(str(addtl_choices[i]).split(" ", 2)[1])
        file.write('\n')
    for i in range(9, 16):
        file.write(str(i))
        file.write(': ')
        file.write(str(addtl_choices[i][0]).split(" ", 2)[1])
        file.write(', ')
        file.write(str(addtl_choices[i][1]).split(" ", 2)[1])
        file.write('\n')
    file.close()

def writecfg_den(outdir, version, strength, thirdstrength):
    completeName = os.path.abspath(outdir + "/ConfigSettings_v" + str(version) +  ".txt")
    file = open(completeName, "w")
    file.write('Secondary augmentation strength:')
    file.write('-1 \n')
    file.write('Tertiary augmentation strength:')
    file.write(thirdstrength + '\n')
    file.write('1-16: denoised and enhanced \n')
    file.close()

