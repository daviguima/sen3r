class DefaultDicts:

    wfr_l2_bnames = {'B1-400': 'Oa01: 400 nm',
                     'B2-412.5': 'Oa02: 412.5 nm',
                     'B3-442.5': 'Oa03: 442.5 nm',
                     'B4-490': 'Oa04: 490 nm',
                     'B5-510': 'Oa05: 510 nm',
                     'B6-560': 'Oa06: 560 nm',
                     'B7-620': 'Oa07: 620 nm',
                     'B8-665': 'Oa08: 665 nm',
                     'B9-673.75': 'Oa09: 673.75 nm',
                     'B10-681.25': 'Oa10: 681.25 nm',
                     'B11-708.75': 'Oa11: 708.75 nm',
                     'B12-753.75': 'Oa12: 753.75 nm',
                     'B16-778.75': 'Oa16: 778.75 nm',
                     'B17-865': 'Oa17: 865 nm',
                     'B18-885': 'Oa18: 885 nm',
                     'B21-1020': 'Oa21: 1020 nm'}

    wfr_int_flags = [1, 2, 4, 8, 8388608, 16777216, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                     32768, 65536, 131072, 262144, 524288, 2097152, 33554432, 67108864, 134217728, 268435456,
                     4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944,
                     549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416,
                     35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312,
                     1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992,
                     18014398509481984, 36028797018963968]

    wfr_str_flags = ['INVALID', 'WATER', 'LAND', 'CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'SNOW_ICE',
                     'INLAND_WATER', 'TIDAL', 'COSMETIC', 'SUSPECT', 'HISOLZEN', 'SATURATED', 'MEGLINT', 'HIGHGLINT',
                     'WHITECAPS', 'ADJAC', 'WV_FAIL', 'PAR_FAIL', 'AC_FAIL', 'OC4ME_FAIL', 'OCNN_FAIL', 'KDM_FAIL',
                     'BPAC_ON', 'WHITE_SCATT', 'LOWRW', 'HIGHRW', 'ANNOT_ANGSTROM', 'ANNOT_AERO_B', 'ANNOT_ABSO_D',
                     'ANNOT_ACLIM', 'ANNOT_ABSOA', 'ANNOT_MIXR1', 'ANNOT_DROUT', 'ANNOT_TAU06', 'RWNEG_O1', 'RWNEG_O2',
                     'RWNEG_O3', 'RWNEG_O4', 'RWNEG_O5', 'RWNEG_O6', 'RWNEG_O7', 'RWNEG_O8', 'RWNEG_O9', 'RWNEG_O10',
                     'RWNEG_O11', 'RWNEG_O12', 'RWNEG_O16', 'RWNEG_O17', 'RWNEG_O18', 'RWNEG_O21']

    # https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-3-olci/level-2/quality-and-science-flags-op
    wfr_bin2flags = {0: 'INVALID',
                     1: 'WATER',
                     2: 'LAND',
                     3: 'CLOUD',
                     23: 'CLOUD_AMBIGUOUS',
                     24: 'CLOUD_MARGIN',
                     4: 'SNOW_ICE',
                     5: 'INLAND_WATER',
                     6: 'TIDAL',
                     7: 'COSMETIC',
                     8: 'SUSPECT',
                     9: 'HISOLZEN',
                     10: 'SATURATED',
                     11: 'MEGLINT',
                     12: 'HIGHGLINT',
                     13: 'WHITECAPS',
                     14: 'ADJAC',
                     15: 'WV_FAIL',
                     16: 'PAR_FAIL',
                     17: 'AC_FAIL',
                     18: 'OC4ME_FAIL',
                     19: 'OCNN_FAIL',
                     21: 'KDM_FAIL',
                     25: 'BPAC_ON',
                     26: 'WHITE_SCATT',
                     27: 'LOWRW',
                     28: 'HIGHRW',
                     32: 'ANNOT_ANGSTROM',
                     33: 'ANNOT_AERO_B',
                     34: 'ANNOT_ABSO_D',
                     35: 'ANNOT_ACLIM',
                     36: 'ANNOT_ABSOA',
                     37: 'ANNOT_MIXR1',
                     38: 'ANNOT_DROUT',
                     39: 'ANNOT_TAU06',
                     40: 'RWNEG_O1',
                     41: 'RWNEG_O2',
                     42: 'RWNEG_O3',
                     43: 'RWNEG_O4',
                     44: 'RWNEG_O5',
                     45: 'RWNEG_O6',
                     46: 'RWNEG_O7',
                     47: 'RWNEG_O8',
                     48: 'RWNEG_O9',
                     49: 'RWNEG_O10',
                     50: 'RWNEG_O11',
                     51: 'RWNEG_O12',
                     52: 'RWNEG_O16',
                     53: 'RWNEG_O17',
                     54: 'RWNEG_O18',
                     55: 'RWNEG_O21'}

    # WFR pixels with these flags will be removed:
    wfr_remove = ['INVALID', 'CLOUD', 'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'SNOW_ICE',
                  'SUSPECT', 'SATURATED', 'AC_FAIL', 'MEGLINT', 'HIGHGLINT', 'LOWRW']

    # Pixels must include these flags:
    wfr_keep = ['INLAND_WATER']

    # ,---------------------,
    # | NetCDF engine dicts |-------------------------------------------------------------------------------------------
    # '---------------------'

    # TODO: used in parallel only (verify).
    wfr_files_p = (('w_aer.nc', 'A865'),
                   ('w_aer.nc', 'T865'),
                   ('Oa01_reflectance.nc', 'Oa01_reflectance'),
                   ('Oa02_reflectance.nc', 'Oa02_reflectance'),
                   ('Oa03_reflectance.nc', 'Oa03_reflectance'),
                   ('Oa04_reflectance.nc', 'Oa04_reflectance'),
                   ('Oa05_reflectance.nc', 'Oa05_reflectance'),
                   ('Oa06_reflectance.nc', 'Oa06_reflectance'),
                   ('Oa07_reflectance.nc', 'Oa07_reflectance'),
                   ('Oa08_reflectance.nc', 'Oa08_reflectance'),
                   ('Oa09_reflectance.nc', 'Oa09_reflectance'),
                   ('Oa10_reflectance.nc', 'Oa10_reflectance'),
                   ('Oa11_reflectance.nc', 'Oa11_reflectance'),
                   ('Oa12_reflectance.nc', 'Oa12_reflectance'),
                   ('Oa16_reflectance.nc', 'Oa16_reflectance'),
                   ('Oa17_reflectance.nc', 'Oa17_reflectance'),
                   ('Oa18_reflectance.nc', 'Oa18_reflectance'),
                   ('Oa21_reflectance.nc', 'Oa21_reflectance'),
                   ('wqsf.nc', 'WQSF'))

    syn_files = {
        'Syn_AOT550.nc': ['T550'],
        'Syn_Angstrom_exp550.nc': ['A550']
    }

    syn_vld_names = {}  # TODO: populate with the valid Synergy equivalent netcdf files.

    wfr_files = {
        'w_aer.nc': ['A865', 'T865'],
        'Oa01_reflectance.nc': ['Oa01_reflectance'],
        'Oa02_reflectance.nc': ['Oa02_reflectance'],
        'Oa03_reflectance.nc': ['Oa03_reflectance'],
        'Oa04_reflectance.nc': ['Oa04_reflectance'],
        'Oa05_reflectance.nc': ['Oa05_reflectance'],
        'Oa06_reflectance.nc': ['Oa06_reflectance'],
        'Oa07_reflectance.nc': ['Oa07_reflectance'],
        'Oa08_reflectance.nc': ['Oa08_reflectance'],
        'Oa09_reflectance.nc': ['Oa09_reflectance'],
        'Oa10_reflectance.nc': ['Oa10_reflectance'],
        'Oa11_reflectance.nc': ['Oa11_reflectance'],
        'Oa12_reflectance.nc': ['Oa12_reflectance'],
        'Oa16_reflectance.nc': ['Oa16_reflectance'],
        'Oa17_reflectance.nc': ['Oa17_reflectance'],
        'Oa18_reflectance.nc': ['Oa18_reflectance'],
        'Oa21_reflectance.nc': ['Oa21_reflectance'],
        'wqsf.nc': ['WQSF']
    }

    wfr_vld_names = {
        'lon': 'longitude:double',
        'lat': 'latitude:double',
        'Oa01_reflectance': 'Oa01_reflectance:float',
        'Oa02_reflectance': 'Oa02_reflectance:float',
        'Oa03_reflectance': 'Oa03_reflectance:float',
        'Oa04_reflectance': 'Oa04_reflectance:float',
        'Oa05_reflectance': 'Oa05_reflectance:float',
        'Oa06_reflectance': 'Oa06_reflectance:float',
        'Oa07_reflectance': 'Oa07_reflectance:float',
        'Oa08_reflectance': 'Oa08_reflectance:float',
        'Oa09_reflectance': 'Oa09_reflectance:float',
        'Oa10_reflectance': 'Oa10_reflectance:float',
        'Oa11_reflectance': 'Oa11_reflectance:float',
        'Oa12_reflectance': 'Oa12_reflectance:float',
        'Oa16_reflectance': 'Oa16_reflectance:float',
        'Oa17_reflectance': 'Oa17_reflectance:float',
        'Oa18_reflectance': 'Oa18_reflectance:float',
        'Oa21_reflectance': 'Oa21_reflectance:float',
        'OAA': 'OAA:float',
        'OZA': 'OZA:float',
        'SAA': 'SAA:float',
        'SZA': 'SZA:float',
        'WQSF': 'WQSF_lsb:double',
        'A865': 'A865:float',
        'T865': 'T865:float',
    }

    l1_wave_bands = {
        'Oa01': 400,
        'Oa02': 412.5,
        'Oa03': 442.5,
        'Oa04': 490,
        'Oa05': 510,
        'Oa06': 560,
        'Oa07': 620,
        'Oa08': 665,
        'Oa09': 673.75,
        'Oa10': 681.25,
        'Oa11': 708.75,
        'Oa12': 753.75,
        'Oa13': 761.25,
        'Oa14': 764.375,
        'Oa15': 767.5,
        'Oa16': 778.75,
        'Oa17': 865,
        'Oa18': 885,
        'Oa19': 900,
        'Oa20': 940,
        'Oa21': 1020
    }

    s3_bands_l2 = {
        'Oa01': 400,
        'Oa02': 412.5,
        'Oa03': 442.5,
        'Oa04': 490,
        'Oa05': 510,
        'Oa06': 560,
        'Oa07': 620,
        'Oa08': 665,
        'Oa09': 673.75,
        'Oa10': 681.25,
        'Oa11': 708.75,
        'Oa12': 753.75,
        'Oa16': 778.75,
        'Oa17': 865,
        'Oa18': 885,
        'Oa21': 1020
    }

