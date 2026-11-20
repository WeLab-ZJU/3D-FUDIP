import numpy as np
import torch
import gc


def blendZ(overlap_bottom, overlap_top, ramp_start):
    return overlap_bottom * (1 - ramp_start) + overlap_top * ramp_start

def blend2d(dataA, dataB, dim, overlap):
    if dataA.size == 0:
        dataC = dataB
    # stitching in x direction
    elif dim == 2:
        ny = dataA.shape[0]
        overlap_right = dataA[:, -overlap:]
        overlap_left = dataB[:, :overlap]

        ramp = np.linspace(0, 1, overlap)
        ramp = ramp[np.newaxis, :]
        ramp_start = np.tile(ramp, (ny, 1))

        Overlap_X = overlap_right * (1-ramp_start) + overlap_left * ramp_start
        dataC = np.concatenate((dataA[:, :-overlap], Overlap_X, dataB[:, overlap:]), axis=1)

    # stitching in y direction
    elif dim == 1:
        nx = dataA.shape[1]
        overlap_start = dataA[-overlap:, :]
        overlap_end = dataB[:overlap, :]

        ramp = np.linspace(0, 1, overlap)
        ramp = ramp[:, np.newaxis]
        ramp_start = np.tile(ramp, (1, nx))

        Overlap_Y = overlap_start * (1 - ramp_start) + overlap_end * ramp_start
        dataC = np.concatenate((dataA[:-overlap, :], Overlap_Y, dataB[overlap:, :]), axis=0)

    return dataC


def stitchingZ(sub_stacks, CropStepXY, CropStepZ, overlap_xy, overlap_z, cut, tilez, tiley, tilex):
    assert cut > 0, "Boundary size must be greater than 0."
    xSizeF, ySizeF, zSizeF = CropStepXY, CropStepXY, CropStepZ
    xSize, ySize, zSize = xSizeF + overlap_xy, ySizeF + overlap_xy, zSizeF + overlap_z

    overlap = overlap_z - cut * 2
    assert overlap > 0, "Overlap size must be greater than 0."
    output_z = np.empty((sub_stacks.shape[1], sub_stacks.shape[2]), dtype=object)

    # M_size = [(zSize-2*cut)*tilez-overlap*(tilez-1), ySize-2*cut, xSize-2*cut]
    # weight map
    ramp = np.linspace(0, 1, overlap)
    ramp = ramp[:, np.newaxis, np.newaxis]
    # extend 1d weight map to 3d
    # ramp_start = np.tile(ramp, (1, ySize - 2 * cut, xSize - 2 * cut))

    # block_z store the tile information
    block_z = np.empty(tilez, dtype=object)
    print('--------------- start Z stitch -----------------')
    for x in range(tilex): #tilex
        for y in range(tiley): #tiley, 5, 6
            for z in range(tilez):
                # extract all z stack in a specific (x,y) position
                block_z[z] = sub_stacks[z, y, x]

            # Array size
            M_size = [(zSize - 2 * cut) * tilez - overlap * (tilez - 1), ySize - 2 * cut, xSize - 2 * cut]

            # in y direction
            if block_z[tilez-1].shape[1] != ySize:
                M_size[1] = block_z[tilez-1].shape[1]- 2*cut
            # in x direction
            if block_z[tilez-1].shape[2] != xSize:
                M_size[2] = block_z[tilez-1].shape[2]- 2*cut
            # in z direction
            if block_z[tilez-1].shape[0] != block_z[0].shape[0]:
                out_size_z = np.abs(block_z[0].shape[0] - block_z[tilez-1].shape[0])
                M_size[0] = M_size[0] - out_size_z

            if block_z[tilez-1].shape[0] - 2*cut < overlap:
                ramp = np.linspace(0, 1, block_z[tilez-1].shape[0] - 2*cut)
                ramp = ramp[:, np.newaxis, np.newaxis]

            # extend 1d weight map to 3d
            ramp_start = np.tile(ramp, (1, M_size[1], M_size[2]))

            # output size
            MergeZ = np.zeros(M_size, dtype=np.float32)

            # Merging stack in z direction
            overlap_bottom = np.zeros((overlap, M_size[1], M_size[2]), dtype=np.float32)

            for zz in range(tilez):
                stack = block_z[zz].astype(np.float32)
                # crop boundrary
                stack_crop = stack[cut:-cut, cut:-cut, cut:-cut]

                # calculate start and end position in output array
                z_start = zz * (zSize - 2*cut - overlap)
                z_end = min(M_size[0], z_start + zSize - 2*cut)

                # First, put all in
                MergeZ[z_start:z_end, :, :] = stack_crop
                # Merge overlap with up and down
                overlap_top = stack_crop[0:overlap, :, :]
                if zz == 0:
                    overlap_bottom = overlap_top

                if zz == tilez-1 and block_z[tilez-1].shape[0] - 2*cut < overlap:
                    z_last = overlap_top.shape[0]
                    overlap_bottom = overlap_bottom[-z_last:, :, :]
                OverlapZ = blendZ(overlap_bottom, overlap_top, ramp_start)
                # Put merged overlap in
                MergeZ[z_start:z_start+overlap, :, :] = OverlapZ
                # Preparing for next stich
                overlap_bottom = stack_crop[-overlap:, :, :]
                print("Tile_{} has been merged!".format(zz))

            print("Tile_X_{}_Tile_Y_{} has been merged!".format(x, y))
            output_z[y, x] = MergeZ

    print('--------------- End of Z stitch -----------------')

    return output_z


def stitchingXY(blocks, overlap_xy, cut, tiley, tilex):

    overlap = overlap_xy - cut * 2

    flag = 0

    nz = blocks[0][0].shape[0]

    print('--------------- start XY stitch -----------------')
    # Stitching xy plane in every z plane
    for z in range(nz):
        MergeY = np.array([])

        for y in range(tiley):
            MergeX = np.array([])

            for x in range(tilex):
                temp = blocks[y][x]
                slice_ = temp[z, :, :].astype(np.float32)
                MergeX = blend2d(MergeX, slice_, 2, overlap)

            MergeY = blend2d(MergeY, MergeX, 1, overlap)

        if flag == 0:
            out = np.zeros((nz, MergeY.shape[0], MergeY.shape[1]), dtype=np.float32)
            flag = 1

        out[z, :, :] = MergeY

        print('z_direction_{}_have_stitched_done!'.format(z))

    print('------------------------------------------------')
    print('All stitched done!')
    print('------------------------------------------------')

    return out


def _tile_attr(tile_opt, name):
    """Read a tiling option from Namespace or dict."""
    if isinstance(tile_opt, dict):
        return tile_opt[name]
    return getattr(tile_opt, name)


def process_whole_stack(img_norm, model, device, tile_opt, boundary_pixel=0, cut_pixel=1):
    """Run tiled 3D inference with overlap blending and stitching.

    Args:
        img_norm (ndarray): Input volume (D, H, W), already normalized if needed.
        model (nn.Module): Network in eval mode.
        device (torch.device): Inference device.
        tile_opt: Object or dict with img_w, img_h, img_s, gap_w, gap_h, gap_s.
        boundary_pixel (int): Extra border crop after stitching (0 to keep full output).
        cut_pixel (int): Overlap trim size passed to stitchingZ/stitchingXY.
    """
    pad = boundary_pixel + cut_pixel
    pad_width = ((pad, pad), (pad, pad), (pad, pad))
    img_norm = np.pad(img_norm, pad_width, mode='constant', constant_values=0)

    # Cropping image patches
    nz, ny, nx = img_norm.shape

    CropSize_XY = _tile_attr(tile_opt, 'gap_w')
    CropSize_Z = _tile_attr(tile_opt, 'gap_s')

    xSizeF = CropSize_XY
    ySizeF = CropSize_XY
    zSizeF = CropSize_Z

    xSize = _tile_attr(tile_opt, 'img_w')
    ySize = _tile_attr(tile_opt, 'img_h')
    zSize = _tile_attr(tile_opt, 'img_s')

    Overlap_XY = xSize - xSizeF
    Overlap_Z = zSize - zSizeF

    # Number of blocks in each dimension
    block_num_z = int(np.ceil((nz - Overlap_Z) / zSizeF))
    block_num_y = int(np.ceil((ny - Overlap_XY) / ySizeF))
    block_num_x = int(np.ceil((nx - Overlap_XY) / xSizeF))

    # Spliting 3D array into small sub-arrays
    data_blocks = np.empty((block_num_z, block_num_y, block_num_x), dtype=object)

    for num_x in range(block_num_x):
        for num_y in range(block_num_y):
            for num_z in range(block_num_z):
                x_start = xSizeF * num_x
                y_start = ySizeF * num_y
                z_start = zSizeF * num_z

                if x_start >= (nx - Overlap_XY) or y_start >= (ny - Overlap_XY) or z_start >= (nz - Overlap_Z):
                    break

                z_end = z_start + zSize
                y_end = y_start + ySize
                x_end = x_start + xSize

                z_end = min(nz, z_end)
                y_end = min(ny, y_end)
                x_end = min(nx, x_end)

                crop_stack = img_norm[z_start:z_end, y_start:y_end, x_start:x_end]

                array_size = crop_stack.shape
                target_size = (zSize, ySize, xSize)

                if array_size != target_size:
                    padding_values = []
                    for target_s, size_s in zip(target_size, array_size):
                        if target_s == size_s:
                            padding_values.append(0)
                        else:
                            padding_values.append(target_s - (size_s % target_s))

                    padding_new = [(0, value) for value in padding_values]
                    crop_stack = np.pad(crop_stack, padding_new, mode='constant', constant_values=0)

                input_data = torch.from_numpy(crop_stack).unsqueeze(dim=0).unsqueeze(dim=0)
                input_data = input_data.to(device)

                model_out = model(input_data)
                if isinstance(model_out, (list, tuple)):
                    output_res = model_out[-1]
                else:
                    output_res = model_out

                output_res = output_res.squeeze().cpu().detach().numpy()
                if array_size != target_size:
                    output_res = output_res[:target_size[0] - padding_values[0], :target_size[1] - padding_values[1], :target_size[2] - padding_values[2]]
                data_blocks[num_z, num_y, num_x] = output_res

                del model_out, input_data, output_res
                torch.cuda.empty_cache()
                print("Processing block number: z:{}, y:{}, x:{}".format(num_z, num_y, num_x))

    print('--- Start to stitch arrays. ---')
    # stitch cropped data
    del img_norm
    gc.collect()

    merge_z = stitchingZ(data_blocks, CropSize_XY, CropSize_Z, Overlap_XY,
                         Overlap_Z, cut_pixel,
                         block_num_z, block_num_y, block_num_x)
    del data_blocks
    gc.collect()
    outputx = stitchingXY(merge_z, Overlap_XY, cut_pixel, block_num_y, block_num_x)

    del merge_z
    gc.collect()

    outputx[outputx < 0] = 0

    if boundary_pixel > 0:
        outputx = outputx[
            boundary_pixel:-boundary_pixel,
            boundary_pixel:-boundary_pixel,
            boundary_pixel:-boundary_pixel]

    return outputx