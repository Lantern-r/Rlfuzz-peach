import os

from gym.envs.registration import register


def afl_forkserver_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/afl-2.52b-mod/afl-2.52b/afl-forkserver',
    )


def afl_2_57_forkserver_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/afl-2.57b-mod/afl-2.57b/afl-forkserver',
    )


# base64_afl
def base64_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/lava-m-mod/base64_afl',
    )


register(
    id='FuzzBase64-v0',
    entry_point='rlfuzz.envs:FuzzBase64Env',
)


# md5sum_afl
def md5sum_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/lava-m-mod/md5sum_afl',
    )


register(
    id='FuzzMd5sum-v0',
    entry_point='rlfuzz.envs:FuzzMd5sumEnv',
)


# uniq_afl
def uniq_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/lava-m-mod/uniq_afl',
    )


register(
    id='FuzzUniq-v0',
    entry_point='rlfuzz.envs:FuzzUniqEnv',
)


# who_afl
def who_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/lava-m-mod/who_afl',
    )


register(
    id='FuzzWho-v0',
    entry_point='rlfuzz.envs:FuzzWhoEnv',
)


# tar_afl

def gzip_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/gzip-mod/gzip_afl',
    )


register(
    id='Fuzzgzip-v0',
    entry_point='rlfuzz.envs:FuzzgzipEnv',
)


# boringssl

def boringssl_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/fuzzer-test-suite-mod/programs/boringssl-2016-02-12-afl'
    )


register(
    id='Fuzzboringssl-v0',
    entry_point='rlfuzz.envs:FuzzboringsslEnv',
)

# freetype2

def freetype2_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/fuzzer-test-suite-mod/programs/freetype2-2017-afl',
    )


register(
    id='Fuzzfreetype2-v0',
    entry_point='rlfuzz.envs:Fuzzfreetype2Env',
)


# libpng

def libpng_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/fuzzer-test-suite-mod/programs/libpng-1.2.56-afl',
    )


register(
    id='Fuzzlibpng-v0',
    entry_point='rlfuzz.envs:FuzzlibpngEnv',
)


# gueztil

def gueztil_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/fuzzer-test-suite-mod/programs/guetzli-2017-3-30-afl',
    )


register(
    id='Fuzzguetzil-v0',
    entry_point='rlfuzz.envs:FuzzguetzilEnv',
)


# libjpeg

def libjpeg_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/fuzzer-test-suite-mod/programs/libjpeg-turbo-07-2017-afl',
    )


register(
    id='Fuzzlibjpeg-v0',
    entry_point='rlfuzz.envs:FuzzlibjpegEnv',
)


# pngquant
def pngquant_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/pngquant-mod/pngquant-master/pngquant-afl',
    )


register(
    id='FuzzPngquant-v0',
    entry_point='rlfuzz.envs:FuzzpngquantEnv',
)


# CImg
def CImg_target_path():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        package_directory, 'mods/Cimg-mod/image2ascii',
    )


register(
    id='FuzzCImg-v0',
    entry_point='rlfuzz.envs:FuzzCImgEnv',
)

# user defined binary
register(
    id='UserDefined-v0',
    entry_point='rlfuzz.envs:FuzzUserDefinedEnv',
)

register(
    id='FUzzSocket-v0',
    entry_point='rlfuzz.envs:FuzzSocketEnv'
)

register(
    id='FuzzAC68U-v0',
    entry_point='rlfuzz.envs:FuzzAC68UEnv'
)

register(
    id='FuzzAC9-v0',
    entry_point='rlfuzz.envs:FuzzAC9Env'
)
