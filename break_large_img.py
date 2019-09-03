# Defines how to read large images
# Note: Packages define X/Y direction only;
# z_blocks define splitting z direction


# Z-direction splitting

def break_large_img(imagesize):
    if imagesize[2] > 100:
        z_blocks = list(range(0, imagesize[2], 100))
        if z_blocks[-1] < imagesize[2]:
            z_blocks.append(imagesize[2])
        if z_blocks[-1] < z_blocks[-2] + 5:
            temp = z_blocks[:-2]
            temp.append(z_blocks[-1])
            z_blocks = temp
        print('Data will be split in z direction at planes:')
        print(z_blocks)
    else:
        z_blocks = [0, imagesize[2]]

    # Check for image dimensions, if large break in this direction
    if imagesize[0] > 1024:
        x_breaks = list(range(0, imagesize[0], 1000))
        if x_breaks[-1] < imagesize[0]:
            x_breaks.append(imagesize[0])
    else:
        x_breaks = [0, imagesize[0]]

    if imagesize[1] > 1024:
        y_breaks = list(range(0, imagesize[1], 1000))
        if y_breaks[-1] < imagesize[1]:
            y_breaks.append(imagesize[1])
    else:
        y_breaks = [0, imagesize[1]]

    # Define boundaries what to read, with certain overlap
    packs = (len(x_breaks) - 1) * (len(y_breaks) - 1)
    if packs > 1:
        temp_packages_x = [[x_breaks[xx] - 12, x_breaks[xx + 1] + 12]
                           for xx in range(len(x_breaks) - 1)]
        temp_packages_x[0][0] = x_breaks[0]
        temp_packages_x[-1][1] = x_breaks[-1]

        temp_packages_y = [[y_breaks[yy] - 12, y_breaks[yy + 1] + 12]
                           for yy in range(len(y_breaks) - 1)]
        temp_packages_y[0][0] = y_breaks[0]
        temp_packages_y[-1][1] = y_breaks[-1]

        packages = [x + y for x in temp_packages_x for y in temp_packages_y]
    else:
        packages = [[0, imagesize[0], 0, imagesize[1]]]

    return packages, z_blocks
