

# TODO
def calc_outer_coords_aligned(align_transform_size):
    buf = align_transform_size // 4
    return [
        -buf, -buf, # nw
        -buf, align_transform_size + buf, # sw
        align_transform_size + buf, align_transform_size + buf, # se
        align_transform_size + buf, -buf # ne
    ]


DEBUG = False
CNA_SIZE = 1024
ALIGN_TRANSFORM_SIZE = 2048
INPAINT_SIZE = 512
OUTER_INNER_RATIO = 1.5
INNER_IMSIZE = int(INPAINT_SIZE / OUTER_INNER_RATIO)
SEG_PRED_IMSIZE = 192
INNER_SEG_PRED_IMSIZE = int(SEG_PRED_IMSIZE / OUTER_INNER_RATIO)
OUTER_COORDS_ALIGNED = calc_outer_coords_aligned(ALIGN_TRANSFORM_SIZE)
OUTER_CNA_SIZE = int(OUTER_INNER_RATIO * CNA_SIZE)
MAX_MAG = 1.5
#W_LERP_EXP = 'lerp/5/5'
#AE_EXP = 'rec/25/8'
#ENC_LERP_EXP = 'enc-lerp/1/2'
#OUTER_SEG_EXP = 'outer-seg/0/1'
FT_N_MAGS = 8
